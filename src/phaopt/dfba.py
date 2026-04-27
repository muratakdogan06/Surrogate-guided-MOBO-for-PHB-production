"""
phaopt.dfba — Dynamic Flux Balance Analysis (dFBA) for PHA production.

Implements the Static Optimisation Approach (SOA) for dFBA:
    At each time step Δt the LP is solved at steady state (standard FBA),
    then extracellular metabolite concentrations and biomass are updated
    using Euler forward integration of the exchange fluxes.

    dX/dt = μ · X
    dS_i/dt = v_i · X          (v_i < 0 for uptake, > 0 for secretion)

Uptake fluxes are bounded by Michaelis–Menten kinetics so that substrate
limitation emerges naturally as concentrations decrease.

References
----------
Mahadevan, Edwards & Doyle (2002) Biophys J 83:1331-1340
Höffner, Harwood, Barton (2013) Comput Chem Eng 49:85-91
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Substrate:
    """A tracked extracellular metabolite with Michaelis–Menten uptake."""

    name: str
    exchange_rxn_id: str
    initial_conc: float          # mmol/L
    km: float = 0.5             # Michaelis constant (mmol/L)
    vmax: float = 10.0          # max uptake rate (mmol/gDW/h), positive value
    molecular_weight: float = 1.0  # g/mmol, for g/L conversion if needed


@dataclass
class DFBAConfig:
    """Parameters for a single dFBA simulation."""

    t_start: float = 0.0        # h
    t_end: float = 48.0         # h
    dt: float = 0.5             # h
    biomass_init: float = 0.05  # gDW/L
    volume: float = 1.0         # L (constant, batch reactor)
    substrates: List[Substrate] = field(default_factory=list)
    biomass_rxn_id: str = "biomass"
    pha_rxn_id: str = "DM_POLHYBU_c"
    pha_init: float = 0.0       # g/L accumulated PHA
    pha_mw: float = 86.09       # g/mol monomer (3-hydroxybutyrate)
    # Extra exchange overrides (non-substrate, e.g. O2 limit)
    static_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class DFBAResult:
    """Full time-course output of a dFBA run."""

    time: np.ndarray
    biomass: np.ndarray          # gDW/L
    pha_accumulated: np.ndarray  # mmol/L
    growth_rate: np.ndarray      # 1/h
    pha_flux: np.ndarray         # mmol/gDW/h
    substrate_concs: Dict[str, np.ndarray]   # mmol/L per substrate
    substrate_uptake: Dict[str, np.ndarray]  # mmol/gDW/h per substrate
    status: List[str]            # solver status per step

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of the time course."""
        data = {
            "time_h": self.time,
            "biomass_gDW_L": self.biomass,
            "pha_mmol_L": self.pha_accumulated,
            "growth_rate_1_h": self.growth_rate,
            "pha_flux_mmol_gDW_h": self.pha_flux,
            "status": self.status,
        }
        for name, arr in self.substrate_concs.items():
            data[f"{name}_mmol_L"] = arr
        for name, arr in self.substrate_uptake.items():
            data[f"{name}_uptake_mmol_gDW_h"] = arr
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Core dFBA engine
# ---------------------------------------------------------------------------

def _mm_bound(conc: float, vmax: float, km: float) -> float:
    """Michaelis–Menten uptake bound (returns negative lower bound)."""
    if conc <= 0.0:
        return 0.0
    return -vmax * conc / (km + conc)


def run_dfba(
    model,
    config: DFBAConfig,
    design: Optional[Dict[str, Any]] = None,
    upreg_fold: float = 3.0,
) -> DFBAResult:
    """
    Run dynamic FBA using the Static Optimisation Approach.

    Parameters
    ----------
    model : cobra.Model
        The extended GEM (with PHA pathway).
    config : DFBAConfig
        Simulation parameters including substrates and time span.
    design : dict, optional
        ``{"knockouts": [...], "upregulations": [...]}``
        If None, wild-type simulation is performed.
    upreg_fold : float
        Fold-change multiplier for upregulated reactions.

    Returns
    -------
    DFBAResult
    """
    from phaopt.simulation import _apply_overrides, _apply_design

    n_steps = int(np.ceil((config.t_end - config.t_start) / config.dt)) + 1
    times = np.linspace(config.t_start, config.t_end, n_steps)

    # Initialise state vectors
    biomass = np.zeros(n_steps)
    pha_acc = np.zeros(n_steps)
    mu_arr = np.zeros(n_steps)
    pha_flux_arr = np.zeros(n_steps)
    sub_concs = {s.name: np.zeros(n_steps) for s in config.substrates}
    sub_uptake = {s.name: np.zeros(n_steps) for s in config.substrates}
    statuses: List[str] = [""] * n_steps

    # Initial conditions
    biomass[0] = config.biomass_init
    pha_acc[0] = config.pha_init
    for s in config.substrates:
        sub_concs[s.name][0] = s.initial_conc

    rxn_ids = {r.id for r in model.reactions}

    for i in range(n_steps - 1):
        dt = times[i + 1] - times[i]

        with model as m:
            # 1. Apply static overrides (e.g. O2 limit)
            _apply_overrides(m, config.static_overrides)

            # 2. Apply genetic design
            if design is not None:
                _apply_design(m, design, upreg_fold, candidates_cfg={})

            # 3. Set Michaelis–Menten uptake bounds for each substrate
            for s in config.substrates:
                if s.exchange_rxn_id in rxn_ids:
                    lb = _mm_bound(sub_concs[s.name][i], s.vmax, s.km)
                    rxn = m.reactions.get_by_id(s.exchange_rxn_id)
                    rxn.lower_bound = lb
                    # Allow secretion (upper bound stays at model default)

            # 4. Solve FBA — maximise biomass (standard objective)
            m.objective = config.biomass_rxn_id
            sol = m.optimize()

            if sol.status == "optimal" and sol.objective_value > 1e-9:
                mu = sol.objective_value
                v_pha = sol.fluxes.get(config.pha_rxn_id, 0.0)
                statuses[i] = "optimal"
            else:
                mu = 0.0
                v_pha = 0.0
                statuses[i] = sol.status if sol else "infeasible"

            mu_arr[i] = mu
            pha_flux_arr[i] = v_pha

            # Record substrate uptake fluxes
            for s in config.substrates:
                if s.exchange_rxn_id in rxn_ids and sol.status == "optimal":
                    sub_uptake[s.name][i] = sol.fluxes.get(s.exchange_rxn_id, 0.0)

        # 5. Euler forward integration
        X = biomass[i]
        biomass[i + 1] = X + mu * X * dt

        # PHA accumulates (intracellular, but tracked as mmol/L culture)
        pha_acc[i + 1] = pha_acc[i] + v_pha * X * dt

        # Substrate depletion
        for s in config.substrates:
            v_s = sub_uptake[s.name][i]  # negative = uptake
            new_conc = sub_concs[s.name][i] + v_s * X * dt
            sub_concs[s.name][i + 1] = max(new_conc, 0.0)

    # Fill last step status
    statuses[-1] = statuses[-2] if n_steps > 1 else ""

    return DFBAResult(
        time=times,
        biomass=biomass,
        pha_accumulated=pha_acc,
        growth_rate=mu_arr,
        pha_flux=pha_flux_arr,
        substrate_concs=sub_concs,
        substrate_uptake=sub_uptake,
        status=statuses,
    )


# ---------------------------------------------------------------------------
# Batch comparison helper
# ---------------------------------------------------------------------------

def run_dfba_batch(
    model,
    config: DFBAConfig,
    designs: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    upreg_fold: float = 3.0,
) -> Dict[str, DFBAResult]:
    """
    Run dFBA for multiple designs (+ wild-type) and return labelled results.

    Parameters
    ----------
    model : cobra.Model
    config : DFBAConfig
    designs : list[dict]
        Each dict has ``knockouts`` and ``upregulations`` lists.
    labels : list[str], optional
        Human-readable labels; defaults to "Design 1", "Design 2", ...
    upreg_fold : float

    Returns
    -------
    dict[str, DFBAResult]
        Keyed by label ("Wild-type", "Design 1", ...).
    """
    if labels is None:
        labels = [f"Design {i + 1}" for i in range(len(designs))]

    results: Dict[str, DFBAResult] = {}

    # Wild-type first
    logger.info("dFBA: running wild-type ...")
    results["Wild-type"] = run_dfba(model, config, design=None, upreg_fold=upreg_fold)

    for label, design in zip(labels, designs):
        logger.info("dFBA: running %s ...", label)
        results[label] = run_dfba(model, config, design=design, upreg_fold=upreg_fold)

    return results


# ---------------------------------------------------------------------------
# Publication-grade plotting
# ---------------------------------------------------------------------------

def plot_dfba_timecourse(
    results: Dict[str, DFBAResult],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> None:
    """
    4-panel figure: biomass, substrate(s), PHA accumulation, growth & PHA flux.

    Parameters
    ----------
    results : dict[str, DFBAResult]
        Output of ``run_dfba_batch`` (label -> DFBAResult).
    save_path : str, optional
        If given, save figure to this path (PNG, 300 dpi).
    figsize : tuple
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for idx, (label, res) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        lw = 2.5 if label == "Wild-type" else 1.8

        # Panel A: Biomass
        axes[0, 0].plot(res.time, res.biomass, color=c, ls=ls, lw=lw, label=label)

        # Panel B: Substrate concentrations
        for sname, conc in res.substrate_concs.items():
            axes[0, 1].plot(
                res.time, conc, color=c, ls=ls, lw=lw,
                label=f"{label} ({sname})" if idx == 0 or len(results) <= 3 else None,
            )

        # Panel C: PHA accumulation
        axes[1, 0].plot(res.time, res.pha_accumulated, color=c, ls=ls, lw=lw, label=label)

        # Panel D: Growth rate and PHA flux
        axes[1, 1].plot(res.time, res.growth_rate, color=c, ls=ls, lw=lw, label=f"{label} (μ)")
        axes[1, 1].plot(
            res.time, res.pha_flux, color=c, ls=":", lw=lw * 0.8,
            label=f"{label} (PHA)" if idx == 0 else None,
        )

    axes[0, 0].set(xlabel="Time (h)", ylabel="Biomass (gDW/L)", title="(A) Biomass")
    axes[0, 1].set(xlabel="Time (h)", ylabel="Concentration (mmol/L)", title="(B) Substrate")
    axes[1, 0].set(xlabel="Time (h)", ylabel="PHA (mmol/L)", title="(C) PHA Accumulation")
    axes[1, 1].set(xlabel="Time (h)", ylabel="Flux (mmol/gDW/h or 1/h)", title="(D) Growth Rate & PHA Flux")

    for ax in axes.flat:
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    fig.suptitle("Dynamic FBA — PHA Production Time Course", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved dFBA figure: %s", save_path)
        plt.close(fig)
    else:
        plt.close(fig)


def plot_pha_yield_comparison(
    results: Dict[str, DFBAResult],
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing final PHA titre and PHA/biomass yield across designs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    final_pha = [res.pha_accumulated[-1] for res in results.values()]
    final_biomass = [res.biomass[-1] for res in results.values()]
    pha_yield = [
        p / b if b > 1e-6 else 0.0 for p, b in zip(final_pha, final_biomass)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(labels))

    ax1.bar(x, final_pha, color="teal", edgecolor="black", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Final PHA (mmol/L)")
    ax1.set_title("(A) PHA Titre at End of Batch")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, pha_yield, color="coral", edgecolor="black", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("PHA / Biomass (mmol/gDW)")
    ax2.set_title("(B) Specific PHA Yield")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("dFBA — PHA Production Comparison", fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved dFBA yield comparison: %s", save_path)
        plt.close(fig)
    else:
        plt.close(fig)

"""
OptKnock comparison for Reviewer 2 response.
Standalone script — does not modify the main pipeline.
"""

import cobra
import pandas as pd
from itertools import combinations
import time
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# EDIT THESE THREE LINES TO MATCH YOUR PROJECT
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = "/Users/muratakdogan/Desktop/BioresourceTech-article/models/model_with_PHA.xml"  # your SBML with DM_POLHYBU_c
BIOMASS_RXN = "biomass"                            # your biomass reaction ID
PHB_RXN = "DM_POLHYBU_c"                          # your PHB demand reaction ID

# Your 59 flux-active candidate reactions — paste from your YAML config
CANDIDATE_RXNS = [
    "bm00403", "bm00387", "bm00401", "bm00402", "bm00499", "bm00500", "bm00501", "bm00017",
    "bm00279", "bm00397", "bm00429", "bm00374", "bm00375", "bm00376", "bm00377", "bm00390",
    "bm00395", "bm00277", "bm00281", "bm00282", "bm00283", "bm00292", "bm00293", "bm00370",
    "bm00372", "bm00373", "bm00471", "bm00472", "bm00473", "bm00474", "bm00475", "bm00476", 
    "bm00477", "bm00478", "bm00479", "bm00480", "bm00481", "bm00482", "bm00483", "bm00484", 
    "bm00485", "bm00486", "bm00487", "bm00488", "bm00489", "bm00490", "bm00491", "bm00492",
    "bm00493", "bm00494", "bm00495", "bm00496", "bm00497", "bm00498", "bm00296", "bm00298",
    "bm00286", "bm01162", "bm00010"
    # ... add all remaining reaction IDs from your 59-reaction pool
    # load from YAML if easier (see alternative below)
]

# ═══════════════════════════════════════════════════════════════
# ALTERNATIVE: load candidate reactions from your YAML config
# ═══════════════════════════════════════════════════════════════
#import yaml
#with open("configs/candidate_reactions.yaml") as f:
     #cfg = yaml.safe_load(f)
#CANDIDATE_RXNS = cfg["candidate_reactions"]  

EPSILON = 0.30  # same as your active learning threshold


# ═══════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def load_model():
    """Load model and verify essential reactions."""
    model = cobra.io.read_sbml_model(MODEL_PATH)

    # Verify
    rxn_ids = [r.id for r in model.reactions]
    assert BIOMASS_RXN in rxn_ids, f"{BIOMASS_RXN} not found"
    assert PHB_RXN in rxn_ids, f"{PHB_RXN} not found"

    for rid in CANDIDATE_RXNS:
        assert rid in rxn_ids, f"Candidate {rid} not found in model"

    print(f"Model loaded: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites")
    return model


def get_wild_type_baseline(model, epsilon):
    """Get wild-type PHB flux at given epsilon (same as your Section 2.5)."""
    with model:
        # Step 1: max biomass
        model.objective = BIOMASS_RXN
        sol = model.optimize()
        max_bio = sol.objective_value
        print(f"Wild-type max biomass: {max_bio:.4f} h-1")

        # Step 2: constrain biomass, maximize PHB
        model.reactions.get_by_id(BIOMASS_RXN).lower_bound = epsilon * max_bio
        model.objective = PHB_RXN
        sol = model.optimize()
        wt_phb = sol.objective_value
        obs_bio = sol.fluxes[BIOMASS_RXN]
        print(f"Wild-type PHB flux at eps={epsilon}: {wt_phb:.4f} mmol/gDW/h")
        print(f"Wild-type biomass at eps={epsilon}: {obs_bio:.4f} h-1")

    return wt_phb, max_bio


def evaluate_knockout(model, knockouts, epsilon, wt_max_biomass):
    """
    Epsilon-constraint evaluation — identical to your Section 2.5 procedure.
    Returns (phb_flux, biomass_flux) or (0, 0) if lethal.
    """
    with model:
        # Apply knockouts
        for ko in knockouts:
            model.reactions.get_by_id(ko).knock_out()

        # Step 1: check if growth is still possible
        model.objective = BIOMASS_RXN
        sol_bio = model.optimize()

        if sol_bio.status != "optimal" or sol_bio.objective_value < 1e-6:
            return 0.0, 0.0  # lethal

        ko_max_bio = sol_bio.objective_value

        # Step 2: constrain biomass to epsilon * WT max, maximize PHB
        bio_lb = epsilon * wt_max_biomass
        if ko_max_bio < bio_lb:
            return 0.0, 0.0  # cannot sustain required growth

        model.reactions.get_by_id(BIOMASS_RXN).lower_bound = bio_lb
        model.objective = PHB_RXN
        sol_phb = model.optimize()

        if sol_phb.status != "optimal":
            return 0.0, 0.0

        return sol_phb.objective_value, sol_phb.fluxes[BIOMASS_RXN]


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def main():
    model = load_model()
    wt_phb, wt_max_bio = get_wild_type_baseline(model, EPSILON)

    # ─── PHASE 1: Single knockout screen (all 59 reactions) ───
    print("\n" + "=" * 60)
    print("PHASE 1: Single knockout screen (59 reactions)")
    print("=" * 60)

    single_results = []
    t0 = time.time()

    for i, rid in enumerate(CANDIDATE_RXNS):
        phb, bio = evaluate_knockout(model, [rid], EPSILON, wt_max_bio)
        delta = phb - wt_phb
        single_results.append({
            "reaction": rid,
            "phb_flux": round(phb, 4),
            "biomass": round(bio, 4),
            "delta_phb": round(delta, 4),
            "growth_coupled": phb > 0 and bio > 0,
            "lethal": phb == 0 and bio == 0
        })
        if (i + 1) % 10 == 0:
            print(f"  Screened {i + 1}/{len(CANDIDATE_RXNS)}...")

    t1 = time.time()
    df1 = pd.DataFrame(single_results)
    df1 = df1.sort_values("phb_flux", ascending=False).reset_index(drop=True)

    print(f"\nCompleted in {t1 - t0:.1f}s")
    print(f"Lethal knockouts: {df1['lethal'].sum()}")
    print(f"Growth-coupled: {df1['growth_coupled'].sum()}")
    print(f"\nTop 15 single knockouts by PHB flux:")
    print(df1.head(15).to_string(index=False))

    # Check fadA specifically
    fada_row = df1[df1["reaction"] == "bm00377"]
    print(f"\n>>> fadA (bm00377) result:")
    print(f"    PHB flux: {fada_row['phb_flux'].values[0]}")
    print(f"    Biomass:  {fada_row['biomass'].values[0]}")
    print(f"    Rank:     {fada_row.index[0] + 1} / {len(df1)}")

    # ─── PHASE 2: Pairwise knockout screen (top 10 targets) ──
    print("\n" + "=" * 60)
    print("PHASE 2: Pairwise knockout screen (top 10 from Phase 1)")
    print("=" * 60)

    # Exclude lethal single KOs (PhaC, PhaB) from pairing
    viable = df1[df1["growth_coupled"]].head(10)["reaction"].tolist()
    print(f"Top 10 viable targets: {viable}")

    double_results = []
    pairs = list(combinations(viable, 2))
    t0 = time.time()

    for i, (ko1, ko2) in enumerate(pairs):
        phb, bio = evaluate_knockout(model, [ko1, ko2], EPSILON, wt_max_bio)
        double_results.append({
            "knockout_1": ko1,
            "knockout_2": ko2,
            "phb_flux": round(phb, 4),
            "biomass": round(bio, 4),
            "has_fadA": "bm00377" in (ko1, ko2)
        })
        if (i + 1) % 10 == 0:
            print(f"  Screened {i + 1}/{len(pairs)}...")

    t1 = time.time()
    df2 = pd.DataFrame(double_results)
    df2 = df2.sort_values("phb_flux", ascending=False).reset_index(drop=True)

    print(f"\nCompleted in {t1 - t0:.1f}s")
    print(f"\nTop 10 pairwise knockouts:")
    print(df2.head(10).to_string(index=False))

    # ─── PHASE 3: Compare with your MC-EHVI results ──────────
    print("\n" + "=" * 60)
    print("PHASE 3: Comparison summary")
    print("=" * 60)

    best_single = df1.iloc[0]
    best_double = df2.iloc[0]

    print(f"\n{'Method':<30} {'PHB flux':<12} {'Biomass':<10} {'fadA?'}")
    print("-" * 65)
    print(f"{'Wild-type (eps=0.30)':<30} {wt_phb:<12.4f} {'N/A':<10} {'N/A'}")
    print(f"{'Best single KO':<30} {best_single['phb_flux']:<12.4f} "
          f"{best_single['biomass']:<10.4f} "
          f"{'Yes' if best_single['reaction'] == 'bm00377' else 'No'}")
    print(f"{'Best double KO':<30} {best_double['phb_flux']:<12.4f} "
          f"{best_double['biomass']:<10.4f} "
          f"{'Yes' if best_double['has_fadA'] else 'No'}")
    print(f"{'MC-EHVI (this study)':<30} {'2.1635':<12} {'0.2352':<10} {'Yes (5/5)'}")

    # ─── SAVE RESULTS ─────────────────────────────────────────
    df1.to_csv("optknock_single_ko_results.csv", index=False)
    df2.to_csv("optknock_double_ko_results.csv", index=False)
    print("\nResults saved to optknock_single_ko_results.csv")
    print("                   optknock_double_ko_results.csv")


if __name__ == "__main__":
    main()
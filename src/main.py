# main.py
# Priyanshu Luhar

from pathlib import Path
import hashlib

from structures import Structures
from defects import Defects

# -------------------------
# Paths
# -------------------------
input_xyz = Path("../data/original/7200_si_slab.xyz")
doped_dir = Path("../data/doped")
defects_dir = Path("../data/defects")

doped_dir.mkdir(parents=True, exist_ok=True)
defects_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Doping configuration
# -------------------------
dopant_concentrations = [
    (0.0005, 100),  # 0.05% -> 100 structures
    (0.0010, 100),  # 0.1%  -> 100 structures
    (0.0050, 100),  # 0.5%  -> 100 structures
    (0.0100, 100),  # 1.0%  -> 100 structures
]

starting_structure_id = 0

MAKE_DOPE_STRUCTURES = False
MAKE_DEFECT_STRUCTURES = True

# -------------------------
# Deterministic seed helper
# -------------------------
def stable_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    # 32-bit deterministic seed
    return int.from_bytes(hashlib.sha256(s).digest()[:4], "big")


def main() -> None:
    # =========================
    # Stage 1: Make doped structures
    # =========================
    if MAKE_DOPE_STRUCTURES:
        slab = Structures(input_xyz, structure_id=starting_structure_id)

        # Ag doping
        for conc, n_structures in dopant_concentrations:
            for i in range(n_structures):
                slab.dope(
                    dopant="Ag",
                    concentration=conc,
                    seed=stable_seed("DOPE", "Ag", conc, i),
                    out_dir=doped_dir,
                )

        # Au doping
        for conc, n_structures in dopant_concentrations:
            for i in range(n_structures):
                slab.dope(
                    dopant="Au",
                    concentration=conc,
                    seed=stable_seed("DOPE", "Au", conc, i),
                    out_dir=doped_dir,
                )

        # Cu doping
        for conc, n_structures in dopant_concentrations:
            for i in range(n_structures):
                slab.dope(
                    dopant="Cu",
                    concentration=conc,
                    seed=stable_seed("DOPE", "Cu", conc, i),
                    out_dir=doped_dir,
                )

        print(f"Finished generating {slab.structure_id} doped structures in: {doped_dir.resolve()}")

    # =========================
    # Stage 2: Make defect structures
    # =========================
    if MAKE_DEFECT_STRUCTURES:
        doped_files = sorted(doped_dir.glob("Si-*-*.xyz"))
        if not doped_files:
            raise FileNotFoundError(
                f"No doped XYZ files found in {doped_dir}. "
                f"Expected files like Si-Ag-1.xyz, Si-Au-401.xyz, etc."
            )

        # Keep a single global ID across all defect outputs
        defects = Defects(doped_files[0], structure_id=starting_structure_id)

        total_written = 0

        for f in doped_files:
            # reload file into the same Defects instance (keeps structure_id counting upward)
            defects.load_xyz(f)

            # infer a base name from file (e.g., Si-Ag-123)
            base_name = f.stem  # without .xyz

            # Create defect variants for each concentration
            for conc, _n in dopant_concentrations:
                # 1) Schottky (vacancies)
                defects.schottky_defect(
                    concentration=conc,
                    seed=stable_seed("SCHOTTKY", base_name, conc),
                    out_dir=defects_dir,
                    base_name=base_name,
                )
                total_written += 1

                # 2) Frenkel (vacancy + interstitial by moving atoms)
                defects.frenkel_defect(
                    concentration=conc,
                    seed=stable_seed("FRENKEL", base_name, conc),
                    out_dir=defects_dir,
                    base_name=base_name,
                    min_distance=1.8,   # tune if placements fail
                    padding=0.5,
                    max_tries=5000,
                )
                total_written += 1

        print(f"Finished generating {total_written} defect structures in: {defects_dir.resolve()}")
        print(f"Final defect structure_id: {defects.structure_id}")


if __name__ == "__main__":
    main()

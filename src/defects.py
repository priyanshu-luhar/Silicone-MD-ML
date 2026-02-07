# defects.py
# Priyanshu Luhar

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union
import random
import math

Atom = Tuple[str, float, float, float]  # (symbol, x, y, z)


@dataclass
class Defects:
    xyz_path: Union[str, Path]
    structure_id: int = 0
    atoms: List[Atom] = field(default_factory=list)
    comment: str = ""

    def __post_init__(self) -> None:
        self.xyz_path = Path(self.xyz_path)
        self.load_xyz(self.xyz_path)

    def load_xyz(self, path: Union[str, Path]) -> None:
        path = Path(path)
        lines = path.read_text().splitlines()
        if len(lines) < 2:
            raise ValueError(f"XYZ file too short: {path}")

        try:
            n = int(lines[0].strip())
        except Exception as e:
            raise ValueError(f"First line must be atom count in XYZ: {path}") from e

        self.comment = lines[1].rstrip("\n")

        atom_lines = lines[2:]
        if len(atom_lines) < n:
            raise ValueError(f"XYZ says {n} atoms but only {len(atom_lines)} atom lines found: {path}")

        atoms: List[Atom] = []
        for i in range(n):
            parts = atom_lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line {i+3} in {path}: {atom_lines[i]}")
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append((sym, x, y, z))

        self.atoms = atoms

    @staticmethod
    def _normalize_concentration(conc: float) -> float:
        """
        Accepts:
          - fraction in [0,1]
          - percent in (1,100]
        Returns fraction in [0,1].
        """
        if conc < 0:
            raise ValueError("concentration must be non-negative")
        if conc <= 1.0:
            return conc
        if conc <= 100.0:
            return conc / 100.0
        raise ValueError("concentration must be <= 1.0 (fraction) or <= 100.0 (percent)")

    @staticmethod
    def _write_xyz(path: Path, atoms: List[Atom], comment: str = "") -> None:
        lines = [str(len(atoms)), comment]
        for sym, x, y, z in atoms:
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _bounds(atoms: List[Atom]) -> Tuple[float, float, float, float, float, float]:
        xs = [a[1] for a in atoms]
        ys = [a[2] for a in atoms]
        zs = [a[3] for a in atoms]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    @staticmethod
    def _dist2(a: Atom, x: float, y: float, z: float) -> float:
        dx = a[1] - x
        dy = a[2] - y
        dz = a[3] - z
        return dx * dx + dy * dy + dz * dz

    def schottky_defect(
        self,
        concentration: float,
        *,
        seed: Optional[int] = None,
        out_dir: Union[str, Path, None] = None,
        comment: Optional[str] = None,
        base_name: str = "Si",
        even: bool = True,
    ) -> Path:
        """
        Schottky (vacancy) defect in XYZ form:
          - remove a uniform random subset of atoms (vacancies).

        If even=True, we force the number removed to be even (often used for charge-neutral
        Schottky pairs in ionic crystals; harmless here and sometimes desirable).
        """
        frac = self._normalize_concentration(float(concentration))
        n_total = len(self.atoms)
        if n_total == 0:
            raise ValueError("No atoms loaded.")

        rng = random.Random(seed)

        n_remove = int(round(frac * n_total))
        n_remove = max(0, min(n_remove, n_total))

        if even and (n_remove % 2 == 1) and n_remove < n_total:
            n_remove += 1  # make it even
        n_remove = min(n_remove, n_total)

        remove_idx = set(rng.sample(range(n_total), k=n_remove)) if n_remove > 0 else set()
        new_atoms = [a for i, a in enumerate(self.atoms) if i not in remove_idx]

        self.structure_id += 1
        out_dir_path = Path(out_dir) if out_dir is not None else self.xyz_path.parent
        out_dir_path.mkdir(parents=True, exist_ok=True)

        out_path = out_dir_path / f"{base_name}-Schottky-{self.structure_id}.xyz"

        cmt = (comment if comment is not None else self.comment) or ""
        cmt = cmt.strip()
        cmt = f"{cmt} | Schottky vacancies: removed {n_remove}/{n_total} ({frac:.6f})"

        self._write_xyz(out_path, new_atoms, cmt)
        return out_path

    def frenkel_defect(
        self,
        concentration: float,
        *,
        seed: Optional[int] = None,
        out_dir: Union[str, Path, None] = None,
        comment: Optional[str] = None,
        base_name: str = "Si",
        min_distance: float = 1.8,
        padding: float = 0.5,
        max_tries: int = 5000,
    ) -> Path:
        """
        Frenkel defect in XYZ form:
          - select atoms uniformly at random
          - "move" each selected atom to a random interstitial position (uniform in box),
            leaving a vacancy at the original site (implicit) and creating an interstitial.

        Atom count stays the same; coordinates change.
        """
        frac = self._normalize_concentration(float(concentration))
        n_total = len(self.atoms)
        if n_total == 0:
            raise ValueError("No atoms loaded.")

        rng = random.Random(seed)

        n_pairs = int(round(frac * n_total))
        n_pairs = max(0, min(n_pairs, n_total))

        chosen_idx = rng.sample(range(n_total), k=n_pairs) if n_pairs > 0 else []

        # Bounding box for uniform interstitial placement
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds(self.atoms)
        xmin -= padding; xmax += padding
        ymin -= padding; ymax += padding
        zmin -= padding; zmax += padding

        min_d2 = float(min_distance) * float(min_distance)

        # Work on a mutable list
        new_atoms = list(self.atoms)

        # For each chosen atom, find a valid interstitial position
        for idx in chosen_idx:
            sym, _, _, _ = new_atoms[idx]

            placed = False
            for _ in range(max_tries):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                z = rng.uniform(zmin, zmax)

                # ensure not too close to existing atoms
                ok = True
                for a in new_atoms:
                    if self._dist2(a, x, y, z) < min_d2:
                        ok = False
                        break

                if ok:
                    new_atoms[idx] = (sym, x, y, z)  # move atom to interstitial
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Failed to place interstitial for atom index {idx}. "
                    f"Try lowering min_distance or increasing max_tries."
                )

        self.structure_id += 1
        out_dir_path = Path(out_dir) if out_dir is not None else self.xyz_path.parent
        out_dir_path.mkdir(parents=True, exist_ok=True)

        out_path = out_dir_path / f"{base_name}-Frenkel-{self.structure_id}.xyz"

        cmt = (comment if comment is not None else self.comment) or ""
        cmt = cmt.strip()
        cmt = f"{cmt} | Frenkel pairs: moved {n_pairs}/{n_total} atoms ({frac:.6f}), min_dist={min_distance}"

        self._write_xyz(out_path, new_atoms, cmt)
        return out_path

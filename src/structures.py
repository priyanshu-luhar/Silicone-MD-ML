# structures.py
# Priyanshu Luhar

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union
import random

Atom = Tuple[str, float, float, float]  # (symbol, x, y, z)


@dataclass
class Structures:
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

    def dope(
        self,
        dopant: str,
        concentration: float,
        *,
        base_element: str = "Si",
        seed: Optional[int] = None,
        out_dir: Union[str, Path, None] = None,
        comment: Optional[str] = None,
        increment_id: bool = True,
    ) -> Path:
        """
        Uniform random doping: choose a uniform random subset of base_element atoms
        and replace them with dopant.

        Saves to: Si-<dopant>-<ID>.xyz
        """
        dopant = dopant.strip()
        if not dopant:
            raise ValueError("dopant must be a non-empty element string (e.g., 'B', 'P', 'Al')")

        frac = self._normalize_concentration(float(concentration))

        base_indices = [i for i, (sym, _, _, _) in enumerate(self.atoms) if sym == base_element]
        if not base_indices:
            raise ValueError(f"No atoms with symbol '{base_element}' found to dope.")

        n_replace = int(round(frac * len(base_indices)))
        n_replace = max(0, min(n_replace, len(base_indices)))

        rng = random.Random(seed)
        replace_set = set(rng.sample(base_indices, k=n_replace)) if n_replace > 0 else set()

        doped_atoms: List[Atom] = []
        for i, (sym, x, y, z) in enumerate(self.atoms):
            if i in replace_set:
                doped_atoms.append((dopant, x, y, z))
            else:
                doped_atoms.append((sym, x, y, z))

        if increment_id:
            self.structure_id += 1
        curr_id = self.structure_id

        out_dir_path = Path(out_dir) if out_dir is not None else self.xyz_path.parent
        out_dir_path.mkdir(parents=True, exist_ok=True)

        out_path = out_dir_path / f"Si-{dopant}-{curr_id}.xyz"

        cmt = comment if comment is not None else self.comment
        cmt = (cmt or "").strip()
        cmt = f"{cmt} | doped {n_replace}/{len(base_indices)} {base_element}->{dopant} ({frac:.6f})"

        self._write_xyz(out_path, doped_atoms, cmt)
        return out_path

    @staticmethod
    def _write_xyz(path: Path, atoms: List[Atom], comment: str = "") -> None:
        lines = [str(len(atoms)), comment]
        for sym, x, y, z in atoms:
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
        path.write_text("\n".join(lines) + "\n")

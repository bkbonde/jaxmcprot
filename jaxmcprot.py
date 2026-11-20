from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union
import jax
import jax.numpy as jnp
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
import numpy as np


VDW_ELEMENT_PARAMS: dict[str, tuple[float, float]] = {
    "C": (0.1094, 3.3997),
    "N": (0.1700, 3.2500),
    "O": (0.2100, 2.9599),
    "S": (0.2500, 3.5636),
    "P": (0.2000, 3.7418),
    "SE": (0.2500, 3.7418),
    "H": (0.0157, 1.4870),
}
DEFAULT_VDW_PARAMS: tuple[float, float] = (0.1200, 3.4000)


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    res_seq: int
    i_code: str
    res_name: str

    def __str__(self) -> str:
        code = self.i_code.strip()
        seq_str = f"{self.res_seq}{code}" if code else f"{self.res_seq}"
        return f"{self.chain_id}:{seq_str}:{self.res_name}"


def _resolve_atom_element(atom) -> str:
    # Prefer explicit element attribute if available on the Universe
    elem = getattr(atom, "element", "")
    if isinstance(elem, str) and elem.strip():
        cleaned = "".join(ch for ch in elem.strip().upper() if ch.isalpha())
        if cleaned in VDW_ELEMENT_PARAMS:
            return cleaned
        if len(cleaned) >= 2 and cleaned[:2] in VDW_ELEMENT_PARAMS:
            return cleaned[:2]
        if cleaned:
            return cleaned

    # Otherwise use MDAnalysis's atom-name guesser
    guessed = guess_atom_element(atom.name)
    if guessed:
        g_up = str(guessed).strip().upper()
        if g_up in VDW_ELEMENT_PARAMS:
            return g_up
        if len(g_up) >= 2 and g_up[:2] in VDW_ELEMENT_PARAMS:
            return g_up[:2]
        return g_up[:1]

    name_up = atom.name.strip().upper()
    return name_up[0] if name_up else "C"


def _build_residue_metadata(atoms: mda.AtomGroup) -> tuple[np.ndarray, list[ResidueKey]]:
    """Maps MDAnalysis AtomGroup residue indices contiguously to 0..N-1."""
    raw_resindices = atoms.resindices
    unique_raw: list[int] = []
    raw_to_contig: dict[int, int] = {}

    for r_idx in raw_resindices:
        r_int = int(r_idx)
        if r_int not in raw_to_contig:
            raw_to_contig[r_int] = len(unique_raw)
            unique_raw.append(r_int)

    contig_indices = np.array([raw_to_contig[int(r)] for r in raw_resindices], dtype=np.int32)

    residues: list[ResidueKey] = []
    for r_int in unique_raw:
        res = atoms.universe.residues[r_int]
        segid = getattr(res, "segid", "") or "_"
        chain_id = getattr(res, "chainID", "") or segid
        icode = getattr(res, "icode", "") or ""
        res_name = (getattr(res, "resname", "") or "UNK").strip()
        res_seq = int(getattr(res, "resid", r_int + 1))
        residues.append(
            ResidueKey(
                chain_id=str(chain_id).strip() or "_",
                res_seq=res_seq,
                i_code=str(icode),
                res_name=res_name,
            )
        )

    return contig_indices, residues


def parse_pdb_for_mcpath(
    pdb_path: Union[str, Path],
    trajectory_path: Optional[Union[str, Path]] = None,
    selection: str = "protein",
    include_hydrogens: bool = True,
    frame: int = 0,
    vdw_table: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[ResidueKey], mda.Universe]:
    """
    Loads a PDB (and optional trajectory) into an MDAnalysis Universe and returns
    arrays ready for `build_residue_energy_matrix` at the requested frame:

      coords          : (M, 3) float32 array of XYZ coordinates at `frame`
      residue_indices : (M,) int32 array in 0 .. N-1
      epsilon         : (M,) float32 array of well depths (kcal/mol)
      sigma           : (M,) float32 array of collision diameters (Angstroms)
      residues        : list of length N mapping index -> ResidueKey
      universe        : loaded `MDAnalysis.Universe` instance
    """
    table = vdw_table if vdw_table is not None else VDW_ELEMENT_PARAMS

    pdb_str = str(Path(pdb_path))
    if trajectory_path is not None:
        u = mda.Universe(pdb_str, str(Path(trajectory_path)))
    else:
        u = mda.Universe(pdb_str)

    atoms = u.select_atoms(selection)
    if not include_hydrogens:
        atoms = atoms.select_atoms("not name H* and not element H")

    if len(atoms) == 0:
        raise ValueError(
            f"Selection '{selection}' (include_hydrogens={include_hydrogens}) matched 0 atoms in {pdb_str}."
        )

    n_frames = len(u.trajectory)
    if frame < 0:
        frame = n_frames + frame
    if frame < 0 or frame >= n_frames:
        raise IndexError(f"Requested frame {frame} out of range for trajectory of length {n_frames}.")

    u.trajectory[frame]

    coords = np.asarray(atoms.positions, dtype=np.float32)
    residue_indices, residues = _build_residue_metadata(atoms)

    eps_list: list[float] = []
    sig_list: list[float] = []
    for atom in atoms:
        elem = _resolve_atom_element(atom)
        eps, sig = table.get(elem, DEFAULT_VDW_PARAMS)
        eps_list.append(eps)
        sig_list.append(sig)

    epsilon = np.asarray(eps_list, dtype=np.float32)
    sigma = np.asarray(sig_list, dtype=np.float32)

    return coords, residue_indices, epsilon, sigma, residues, u


def compute_trajectory_energy_matrices(
    u: mda.Universe,
    residue_indices: np.ndarray,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    n_residues: int,
    selection: str = "protein",
    include_hydrogens: bool = True,
    frames: Optional[Iterable[int]] = None,
    r_cut: float = 5.5,
) -> jax.Array:
    """
    Iterates over frames in an already-loaded `MDAnalysis.Universe` and computes
    the residue interaction energy matrix for each frame using the JAX function.
    Returns an array of shape (K, n_residues, n_residues).
    """
    atoms = u.select_atoms(selection)
    if not include_hydrogens:
        atoms = atoms.select_atoms("not name H* and not element H")

    res_idx_j = jnp.asarray(residue_indices)
    eps_j = jnp.asarray(epsilon)
    sig_j = jnp.asarray(sigma)

    frame_indices = list(range(len(u.trajectory))) if frames is None else list(frames)
    matrices: list[jax.Array] = []

    for ts_idx in frame_indices:
        u.trajectory[ts_idx]
        frame_coords = jnp.asarray(atoms.positions, dtype=jnp.float32)
        mat = build_residue_energy_matrix(
            frame_coords,
            res_idx_j,
            eps_j,
            sig_j,
            n_residues=n_residues,
            r_cut=r_cut,
        )
        matrices.append(mat)

    return jnp.stack(matrices, axis=0)


def energy_to_transition_matrix(
    res_energy: jax.Array,
    temperature: float = 300.0,
    kb: float = 0.00198720425864083,  # kcal/(mol*K)
) -> jax.Array:
    """Converts an N x N residue energy matrix into a row-normalized Boltzmann matrix."""
    n = res_energy.shape[0]
    beta = 1.0 / (kb * temperature)

    off_diag = 1.0 - jnp.eye(n, dtype=res_energy.dtype)
    shifted = -beta * res_energy
    row_max = jnp.max(jnp.where(off_diag > 0.0, shifted, -jnp.inf), axis=1, keepdims=True)
    weights = jnp.exp(shifted - row_max) * off_diag

    row_sums = jnp.sum(weights, axis=1, keepdims=True)
    uniform_off_diag = off_diag / jnp.maximum(n - 1, 1)

    return jnp.where(
        row_sums > 0.0,
        weights / jnp.maximum(row_sums, 1e-30),
        uniform_off_diag,
    )


@partial(jax.jit, static_argnames=())
def modified_lj_energy(
    r: jax.Array,
    epsilon: jax.Array,
    sigma: jax.Array,
    r_cut: float = 5.5,
    r_min_scale: float = 0.89,
    e_min_cap: float = 10.0,
) -> jax.Array:
    r_safe = jnp.maximum(r, 1e-6)
    lj = 4.0 * epsilon * ((sigma / r_safe) ** 12 - (sigma / r_safe) ** 6)

    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    r_min = r_min_scale * r_eq
    lj = jnp.where(r < r_min, jnp.minimum(lj, e_min_cap), lj)
    lj = jnp.where((r > 0.0) & (r <= r_cut), lj, 0.0)
    return lj


@partial(jax.jit, static_argnames=("n_residues",))
def build_residue_energy_matrix(
    coords: jax.Array,
    residue_indices: jax.Array,
    epsilon: jax.Array,
    sigma: jax.Array,
    n_residues: int,
    r_cut: float = 5.5,
) -> jax.Array:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)

    eps_ij = jnp.sqrt(epsilon[:, None] * epsilon[None, :])
    sig_ij = 0.5 * (sigma[:, None] + sigma[None, :])

    atom_energy = modified_lj_energy(dist, eps_ij, sig_ij, r_cut=r_cut)
    m = atom_energy.shape[0]
    atom_energy = atom_energy * (1.0 - jnp.eye(m, dtype=atom_energy.dtype))

    res_col_sum = jax.vmap(
        lambda row: jax.ops.segment_sum(row, residue_indices, num_segments=n_residues)
    )(atom_energy)

    res_energy = jax.vmap(
        lambda col: jax.ops.segment_sum(col, residue_indices, num_segments=n_residues),
        in_axes=1,
        out_axes=1,
    )(res_col_sum)

    res_energy = res_energy * (1.0 - jnp.eye(n_residues, dtype=res_energy.dtype))
    return res_energy

@partial(jax.jit, static_argnames=("n_steps",))
def _simulate_single_walk(
    prob_matrix: jax.Array,
    key: jax.Array,
    start_res: jax.Array,
    n_steps: int,
) -> jax.Array:
    n_residues = prob_matrix.shape[0]
    step_keys = jax.random.split(key, n_steps)

    def body_fn(curr, step_key):
        next_res = jax.random.categorical(
            step_key, jnp.log(jnp.maximum(prob_matrix[curr], 1e-30))
        )
        return next_res, next_res

    _, trajectory = jax.lax.scan(body_fn, start_res, step_keys)
    return jnp.concatenate([jnp.array([start_res], dtype=trajectory.dtype), trajectory])


_simulate_batch_walks = jax.jit(
    jax.vmap(_simulate_single_walk, in_axes=(None, 0, 0, None)),
    static_argnames=("n_steps",),
)


class JAXMCPathAnalyzer:
    """JAX-accelerated Monte Carlo path generator and centrality calculator."""

    def __init__(
        self,
        prob_matrix: Union[jax.Array, np.ndarray],
        residues: Optional[list[ResidueKey]] = None,
        seed: int = 42,
    ):
        self.prob = jnp.asarray(prob_matrix, dtype=jnp.float32)
        if self.prob.ndim != 2 or self.prob.shape[0] != self.prob.shape[1]:
            raise ValueError("prob_matrix must be a square N x N matrix.")
        self.n_residues = self.prob.shape[0]
        self.residues = residues
        self.key = jax.random.PRNGKey(seed)

    def label(self, idx: int) -> str:
        if self.residues and 0 <= idx < len(self.residues):
            return f"{idx} ({self.residues[idx]})"
        return str(idx)

    def _next_keys(self, count: int) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.split(subkey, count)

    def path_probability(self, path: Union[list[int], tuple[int, ...]]) -> float:
        if len(path) < 2:
            return 1.0
        u = jnp.asarray(path[:-1], dtype=jnp.int32)
        v = jnp.asarray(path[1:], dtype=jnp.int32)
        return float(jnp.prod(self.prob[u, v]))

    def sample_between_residues(
        self,
        start_res: int,
        end_res: int,
        max_steps: int = 12,
        n_samples: int = 10000,
    ) -> list[tuple[tuple[int, ...], int, float]]:
        keys = self._next_keys(n_samples)
        starts = jnp.full((n_samples,), start_res, dtype=jnp.int32)
        trajs = np.asarray(_simulate_batch_walks(self.prob, keys, starts, max_steps))

        counts: Counter[tuple[int, ...]] = Counter()
        for row in trajs:
            hits = np.where(row[1:] == end_res)[0]
            if len(hits) > 0:
                first_hit_idx = int(hits[0]) + 1
                truncated = tuple(int(x) for x in row[: first_hit_idx + 1])
                counts[truncated] += 1

        ranked = [(path, cnt, self.path_probability(path)) for path, cnt in counts.items()]
        ranked.sort(key=lambda item: (-item[1], -item[2]))
        return ranked

    def compute_walk_centrality(
        self,
        walk_length: int = 2000,
        n_walks: int = 50,
    ) -> dict[int, dict[str, float]]:
        keys = self._next_keys(n_walks)
        key_init, key_walks = jax.random.split(self.key)
        self.key = key_init
        starts = jax.random.randint(
            key_walks, shape=(n_walks,), minval=0, maxval=self.n_residues
        )

        walks = np.asarray(_simulate_batch_walks(self.prob, keys, starts, walk_length))
        visit_counts = np.bincount(walks.ravel(), minlength=self.n_residues).astype(float)
        return_intervals: list[list[int]] = [[] for _ in range(self.n_residues)]

        for walk in walks:
            last_seen: dict[int, int] = {}
            for step, res in enumerate(walk):
                r_int = int(res)
                if r_int in last_seen:
                    return_intervals[r_int].append(step - last_seen[r_int])
                last_seen[r_int] = step

        betweenness_proxy = visit_counts / max(float(visit_counts.sum()), 1.0)
        results: dict[int, dict[str, float]] = {}

        for i in range(self.n_residues):
            intervals = return_intervals[i]
            mean_ret = float(np.mean(intervals)) if intervals else float("inf")
            closeness_proxy = (
                (1.0 / mean_ret) if np.isfinite(mean_ret) and mean_ret > 0 else 0.0
            )
            results[i] = {
                "betweenness_proxy": float(betweenness_proxy[i]),
                "closeness_proxy": float(closeness_proxy),
                "mean_return_steps": mean_ret,
            }

        return results


if __name__ == "__main__":
    sample_pdb = Path("1L2Y.pdb")
    if sample_pdb.exists():
        coords_np, res_idx_np, eps_np, sig_np, residues, u = parse_pdb_for_mcpath(
            sample_pdb,
            include_hydrogens=False,
        )
        res_energy = build_residue_energy_matrix(
            jnp.asarray(coords_np),
            jnp.asarray(res_idx_np),
            jnp.asarray(eps_np),
            jnp.asarray(sig_np),
            n_residues=len(residues),
            r_cut=5.5,
        )
        prob = energy_to_transition_matrix(res_energy, temperature=300.0)
        analyzer = JAXMCPathAnalyzer(prob, residues=residues, seed=42)

        centrality = analyzer.compute_walk_centrality(walk_length=2000, n_walks=50)
        top_res = sorted(centrality.items(), key=lambda kv: -kv[1]["betweenness_proxy"])[:5]
        print(f"Parsed {len(residues)} residues from {sample_pdb}:")
        for idx, scores in top_res:
            print(
                f"  {analyzer.label(idx):20s} | betweenness={scores['betweenness_proxy']:.4f} "
                f"| closeness={scores['closeness_proxy']:.4f}"
            )
    else:
        print("Place a PDB file (e.g., 1BE9.pdb) in the working directory and call parse_pdb_for_mcpath().")
        
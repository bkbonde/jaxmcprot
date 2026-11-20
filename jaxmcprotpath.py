from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
import numpy as np


# Standard element-level van der Waals parameters:
# element -> (epsilon in kcal/mol, sigma in Angstroms)
# Representative AMBER / GAFF-derived element-level defaults; not copied from a
# table in PMC3692092.
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
    """Resolves a chemical element symbol from an MDAnalysis Atom object."""
    elem = getattr(atom, "element", "")
    if isinstance(elem, str) and elem.strip():
        cleaned = "".join(ch for ch in elem.strip().upper() if ch.isalpha())
        if cleaned in VDW_ELEMENT_PARAMS:
            return cleaned
        if len(cleaned) >= 2 and cleaned[:2] in VDW_ELEMENT_PARAMS:
            return cleaned[:2]
        if cleaned:
            return cleaned

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

    contig_indices = np.array(
        [raw_to_contig[int(r)] for r in raw_resindices], dtype=np.int32
    )

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


def _iter_universe_frames(u: mda.Universe, frame_indices: list[int]):
    """Steps `u.trajectory` to each requested frame index and yields it."""
    for idx in frame_indices:
        u.trajectory[idx]
        yield idx


def parse_pdb_for_mcpath(
    pdb_path: str | Path,
    trajectory_path: str | Path | None = None,
    selection: str = "protein",
    include_hydrogens: bool = True,
    frame: int = 0,
    vdw_table: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[ResidueKey], mda.Universe]:
    """
    Loads a PDB (and optional trajectory) into an MDAnalysis Universe and returns
    arrays ready for `build_residue_energy_matrix` at the requested frame.
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
        raise IndexError(
            f"Requested frame {frame} out of range for trajectory of length {n_frames}."
        )

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


@partial(jax.jit, static_argnames=())
def modified_lj_energy(
    r: jax.Array,
    epsilon: jax.Array,
    sigma: jax.Array,
    r_cut: float = 5.5,
    r_min_scale: float = 0.89,
    e_min_cap: float = 10.0,
) -> jax.Array:
    """
    Modified Lennard-Jones 12-6 potential for inter-atomic distances r (in Angstroms).
    Caps short-range repulsion to avoid over-penalizing close crystal contacts,
    and truncates interactions beyond r_cut.
    """
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
    """
    Computes the (n_residues, n_residues) residue interaction energy matrix from M atoms.
    """
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


def compute_trajectory_energy_matrices(
    u: mda.Universe,
    residue_indices: np.ndarray,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    n_residues: int,
    selection: str = "protein",
    include_hydrogens: bool = True,
    frames: Iterable[int] | None = None,
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

    for _ in _iter_universe_frames(u, frame_indices):
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


# Alias preserving the name used in the trajectory ensemble step
collect_trajectory_energy_stack = compute_trajectory_energy_matrices


@partial(jax.jit, static_argnames=())
def combine_trajectory_energies(
    energy_stack: jax.Array,
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Combines a stack of residue energy matrices of shape (K, N, N) across K frames
    using normalized non-negative weights (uniform if `weights` is None).
    Returns a combined energy matrix of shape (N, N).
    """
    k = energy_stack.shape[0]
    if weights is None:
        norm_weights = jnp.full((k,), 1.0 / k, dtype=energy_stack.dtype)
    else:
        w = jnp.asarray(weights, dtype=energy_stack.dtype)
        w = jnp.maximum(w, 0.0)
        total_w = jnp.sum(w)
        norm_weights = jnp.where(
            total_w > 0.0,
            w / jnp.maximum(total_w, 1e-30),
            jnp.full((k,), 1.0 / k, dtype=energy_stack.dtype),
        )

    return jnp.einsum("k,kij->ij", norm_weights, energy_stack)


@partial(jax.jit, static_argnames=())
def boltzmann_frame_weights(
    frame_energies: jax.Array,
    temperature: float = 300.0,
    kb: float = 0.00198720425864083,  # kcal/(mol*K)
) -> jax.Array:
    """
    Computes normalized Boltzmann weights w_t proportional to exp(-beta * E_t)
    from a 1D array of per-frame reference/total energies of shape (K,).
    """
    beta = 1.0 / (kb * temperature)
    scaled = -beta * frame_energies
    scaled_shifted = scaled - jnp.max(scaled)
    raw_w = jnp.exp(scaled_shifted)
    return raw_w / jnp.maximum(jnp.sum(raw_w), 1e-30)


@partial(jax.jit, static_argnames=())
def energy_to_transition_matrix(
    res_energy: jax.Array,
    temperature: float = 300.0,
    kb: float = 0.00198720425864083,  # kcal/(mol*K)
) -> jax.Array:
    """
    Converts an N x N residue energy matrix into a row-normalized Boltzmann
    transition probability matrix P where P[i, j] is the probability of stepping
    from residue i to residue j (P[i, i] = 0).
    """
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


def ensemble_transition_matrix_from_universe(
    u: mda.Universe,
    residue_indices: np.ndarray,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    n_residues: int,
    selection: str = "protein",
    include_hydrogens: bool = True,
    frames: Iterable[int] | None = None,
    frame_weights: np.ndarray | jax.Array | None = None,
    use_boltzmann_frame_weights: bool = False,
    temperature: float = 300.0,
    r_cut: float = 5.5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes:
      1. `energy_stack`   : (K, N, N) per-frame residue interaction energy matrices
      2. `combined_energy`: (N, N) weighted average energy matrix across frames
      3. `prob_matrix`    : (N, N) row-normalized Boltzmann transition matrix
    """
    energy_stack = collect_trajectory_energy_stack(
        u=u,
        residue_indices=residue_indices,
        epsilon=epsilon,
        sigma=sigma,
        n_residues=n_residues,
        selection=selection,
        include_hydrogens=include_hydrogens,
        frames=frames,
        r_cut=r_cut,
    )

    if use_boltzmann_frame_weights:
        per_frame_total_energy = jnp.sum(energy_stack, axis=(1, 2))
        weights = boltzmann_frame_weights(per_frame_total_energy, temperature=temperature)
    elif frame_weights is not None:
        weights = jnp.asarray(frame_weights, dtype=energy_stack.dtype)
    else:
        weights = None

    combined_energy = combine_trajectory_energies(energy_stack, weights=weights)
    prob_matrix = energy_to_transition_matrix(combined_energy, temperature=temperature)
    return energy_stack, combined_energy, prob_matrix


@partial(jax.jit, static_argnames=("n_steps",))
def _simulate_single_walk(
    prob_matrix: jax.Array,
    key: jax.Array,
    start_res: jax.Array,
    n_steps: int,
) -> jax.Array:
    """Simulates a walk of length n_steps and returns states of shape (n_steps + 1,)."""
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
        prob_matrix: jax.Array | np.ndarray,
        residues: list[ResidueKey] | None = None,
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

    def path_probability(self, path: list[int] | tuple[int, ...]) -> float:
        """Product of step transition probabilities along a path."""
        if len(path) < 2:
            return 1.0
        u = jnp.asarray(path[:-1], dtype=jnp.int32)
        v = jnp.asarray(path[1:], dtype=jnp.int32)
        return float(jnp.prod(self.prob[u, v]))

    def sample_from_residue(
        self,
        start_res: int,
        n_steps: int,
        n_samples: int = 5000,
    ) -> list[tuple[tuple[int, ...], int, float]]:
        """
        Mode (A): Generates n_samples Monte Carlo paths of length n_steps starting
        from start_res. Returns sorted tuples of (path, count, path_prob).
        """
        keys = self._next_keys(n_samples)
        starts = jnp.full((n_samples,), start_res, dtype=jnp.int32)
        trajs = np.asarray(_simulate_batch_walks(self.prob, keys, starts, n_steps))

        counts: Counter[tuple[int, ...]] = Counter(map(tuple, trajs.tolist()))
        ranked = [(path, cnt, self.path_probability(path)) for path, cnt in counts.items()]
        ranked.sort(key=lambda item: (-item[1], -item[2]))
        return ranked

    def sample_between_residues(
        self,
        start_res: int,
        end_res: int,
        max_steps: int = 12,
        n_samples: int = 10000,
    ) -> list[tuple[tuple[int, ...], int, float]]:
        """
        Mode (B): Samples paths starting at start_res and keeps trajectories that
        reach end_res within max_steps (truncated upon first arrival).
        """
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
        """
        Mode (C): Generates long Monte Carlo walks in JAX and estimates residue
        betweenness-like (visit frequency) and closeness-like (inverse mean
        first-passage / return time) centrality scores.
        """
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
    # Runnable demo on a synthetic 5-frame MDAnalysis Universe
    n_atoms, n_res = 24, 6
    rng = np.random.default_rng(7)
    coords_seq = np.zeros((5, n_atoms, 3), dtype=np.float32)
    for t in range(5):
        coords_seq[t] = rng.normal(loc=t * 0.15, scale=2.2, size=(n_atoms, 3)).astype(np.float32)

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_res,
        atom_resindex=np.repeat(np.arange(n_res), 4),
        trajectory=True,
    )
    for t in range(5):
        if t == 0:
            u.atoms.positions = coords_seq[0]
        else:
            ts = u.trajectory.ts.copy()
            ts.positions = coords_seq[t]
            u.trajectory._ts = ts

    res_idx = np.repeat(np.arange(n_res), 4).astype(np.int32)
    eps = np.full(n_atoms, 0.15, dtype=np.float32)
    sig = np.full(n_atoms, 3.2, dtype=np.float32)

    stack, combined_e, prob = ensemble_transition_matrix_from_universe(
        u=u,
        residue_indices=res_idx,
        epsilon=eps,
        sigma=sig,
        n_residues=n_res,
        selection="all",
        frames=range(len(u.trajectory)),
        temperature=300.0,
    )

    analyzer = JAXMCPathAnalyzer(prob, seed=42)
    print("Top paths from residue 0 to 5:")
    for path, count, p_val in analyzer.sample_between_residues(0, 5, max_steps=10, n_samples=2000)[:5]:
        print(f"  {path} | count={count} | prob={p_val:.3e}")

    print("\nCentrality summary:")
    centrality = analyzer.compute_walk_centrality(walk_length=1000, n_walks=25)
    for res_id, scores in sorted(centrality.items(), key=lambda kv: -kv[1]["betweenness_proxy"]):
        print(
            f"  Residue {res_id}: betweenness={scores['betweenness_proxy']:.4f}, "
            f"closeness={scores['closeness_proxy']:.4f}"
        )
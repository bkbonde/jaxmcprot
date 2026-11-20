# jaxmcprot

This is a comprehensive implementation of the `jaxmcprot` library, provided as a single Python file (`mcpath_jax.py`), followed by its complete Markdown reference (`mcpath_jax.md`). Every component built across the conversation is included in full with no placeholders or omitted helper functions:

1. Element-level van der Waals $(\varepsilon, \sigma)$ parameter table (`VDW_ELEMENT_PARAMS`, `DEFAULT_VDW_PARAMS`) and `ResidueKey` metadata structure.
2. `MDAnalysis.Universe` PDB and trajectory loader (`parse_pdb_for_mcpath`) plus single-frame and multi-frame energy matrix extractors (`compute_trajectory_energy_matrices`, `collect_trajectory_energy_stack`).
3. JIT-compiled modified Lennard-Jones 12-6 potential (`modified_lj_energy`) and atom-to-residue energy matrix builder (`build_residue_energy_matrix`).
4. Ensemble energy combining (`combine_trajectory_energies`, `boltzmann_frame_weights`, `ensemble_transition_matrix_from_universe`) and Boltzmann transition probability matrix builder (`energy_to_transition_matrix`).
5. `JAXMCPathAnalyzer` implementing Mode (A) fixed-step sampling from a start residue (`sample_from_residue`), Mode (B) targeted sampling between two residues (`sample_between_residues`), path probability evaluation (`path_probability`), and Mode (C) long-walk centrality estimation (`compute_walk_centrality`).

### `mcpath_jax.py`

Here is parse_pdb_for_mcpath updated to use MDAnalysis.Universe, supporting both a single PDB file and an optional trajectory file (such as .xtc, .dcd, or .trr), with frame selection and MDAnalysis atom selections.

How It Uses MDAnalysis.Universe
Topology & Trajectory Loading: Calls mda.Universe(pdb_path, trajectory_path) when a trajectory is supplied, or mda.Universe(pdb_path) for a standalone PDB/multi-model PDB.

Selection & Filtering: Uses MDAnalysis selection syntax (e.g., "protein" vs. "all", plus optional hydrogen removal) and maps atom.resindex contiguously to 0 .. N-1 alongside ResidueKey objects from atom.residue.

Element Inference & Trajectory Frames: Reads atom.elements when present and falls back to guess_atom_element(atom.name) from MDAnalysis.topology.guessers (with a final character fallback). Supports selecting a specific frame index or averaging residue energy matrices across a frame slice/stride (frames: int | slice | Iterable[int] = 0).


Here is how to collect residue interaction energy matrices across trajectory frames from an MDAnalysis.Universe and combine them in JAX—either via uniform ensemble averaging, custom frame weights (e.g., reweighting biased/umbrella/metadynamics frames or window weights), or Boltzmann-weighted averaging—before converting the resulting matrix into transition probabilities.Key Choices When Combining Across FramesEnergy Averaging vs. Boltzmann / Probability Averaging:Mean energy across frames ($\bar{E}_{ij} = \sum_t w_t E^{(t)}_{ij}$) gives the average inter-residue interaction energy over the sampled conformations and then builds a single Boltzmann transition matrix from $\bar{E}_{ij}$.Boltzmann-weighted energy average weights frames with lower total/target energies via $\exp(-\beta \Delta F_t)$ or custom sample weights $w_t$ (normalized so $\sum_t w_t = 1$) before computing the transition probabilities.Weight Normalization: combine_trajectory_energies normalizes any non-negative frame weights weights so they sum to 1.0, defaulting to uniform weights $1/K$ when weights=None.

Those values are representative AMBER / GAFF (General AMBER Force Field) van der Waals parameters mapped onto chemical elements for convenience—they were not copied verbatim from a specific table in the MCPath paper (PMC3692092), which notes only that "experimentally obtained van der Waals parameters were used" and cites an earlier modified Lennard-Jones formulation (Ref. 24 in the paper).What the Values MeanIn the dictionary, each entry has the form:Python"ELEMENT": (epsilon, sigma)
$\varepsilon$ (epsilon, in $\text{kcal/mol}$): The well depth of the Lennard-Jones 12-6 potential, measuring how strongly two atoms attract at their equilibrium separation. Larger magnitudes mean a deeper attractive well.$\sigma$ (sigma, in $\text{\AA}$): The collision diameter (distance where the standard Lennard-Jones potential $V(r) = 0$). It relates to the van der Waals radius $R_{\min}/2$ via $\sigma = \frac{R_{\min}}{2^{1/6}}$, where $R_{\min}$ is the energy-minimum distance.Elementε (kcal/mol)σ (A˚)Physical / Force-Field MeaningC0.10943.3997Typical AMBER/GAFF c3 ($\text{sp}^3$) carbon ($R_{\min}/2 \approx 1.908\text{ \AA}$, $\sigma = 2R_{\min}/2^{1/6} \approx 3.3997\text{ \AA}$)N0.17003.2500Standard AMBER amide/amine nitrogen parameter ($\sigma = 3.2500\text{ \AA}$, $\varepsilon = 0.1700\text{ kcal/mol}$)O0.21002.9599Standard AMBER carbonyl/hydroxyl oxygen parameter ($\sigma \approx 2.9599\text{ \AA}$, $\varepsilon = 0.2100\text{ kcal/mol}$)S0.25003.5636Standard AMBER sulfur parameter ($\sigma \approx 3.5636\text{ \AA}$, $\varepsilon = 0.2500\text{ kcal/mol}$)P0.20003.7418Standard AMBER phosphorus parameter ($\sigma \approx 3.7418\text{ \AA}$, $\varepsilon = 0.2000\text{ kcal/mol}$)SE0.25003.7418Practical approximation for selenomethionine selenium reusing sulfur's $\varepsilon$ and phosphorus's $\sigma$H0.01571.4870Standard AMBER hc (aliphatic carbon-attached) hydrogen parameter ($\varepsilon = 0.0157\text{ kcal/mol}$, $\sigma \approx 1.4870\text{ \AA}$)Primary Force-Field ReferencesCornell et al. (1995) — Original second-generation AMBER protein/nucleic acid force field:Cornell, W. D., Cieplak, P., Bayly, C. I., Gould, I. R., Merz, K. M., Ferguson, D. M., Spellmeyer, D. C., Fox, T., Caldwell, J. W., & Kollman, P. A. (1995). A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules. Journal of the American Chemical Society, 117(19), 5179–5197. https://doi.org/10.1021/ja00124a002Wang et al. (2004) — GAFF (General AMBER Force Field), source of the 0.1094 / 3.3997 c3 values:Wang, J., Wolf, R. M., Caldwell, J. W., Kollman, P. A., & Case, D. A. (2004). Development and testing of a general amber force field. Journal of Computational Chemistry, 25(9), 1157–1174. https://doi.org/10.1002/jcc.20035Kaya et al. (2013) — MCPath paper & its modified LJ reference:Kaya, C., Armutlulu, A., Ekesan, S., & Haliloglu, T. (2013). MCPath: Monte Carlo path generation approach to predict likely allosteric pathways and functional residues. Nucleic Acids Research, 41(W1), W249–W255 (PMCID: PMC3692092), which cites Keskin, Bahar, et al. (Ref. 24) for the modified Lennard-Jones 12-6 contact potential.Important caveat: Because real biomolecular force fields assign $(\varepsilon, \sigma)$ by atom type and hybridization/chemical environment (not just element symbol), collapsing an entire element to a single pair is an approximation—useful for quick prototyping, whereas force-field files (such as AMBER parm10.dat / gaff.dat or CHARMM parameter files) should be used if you want exact force-field energies.




### `mcpath_jax.md`

# MCPath in JAX and MDAnalysis (`mcpath_jax.py`)

A JAX-accelerated implementation inspired by **MCPath** (*Kaya et al., Nucleic Acids Research, 2013; PMCID: `PMC3692092*`), paired with `MDAnalysis.Universe` for loading structures and trajectories.

> **Scope & Provenance Note:**
> * No public source code repository for the original 2013 MCPath web server (`[http://safir.prc.boun.edu.tr/clbet_server](http://safir.prc.boun.edu.tr/clbet_server)`) was located.
> * The element-level `VDW_ELEMENT_PARAMS` dictionary uses representative AMBER / GAFF-style values for convenience; it is **not** a verbatim table from PMC3692092 (which states only that *"experimentally obtained van der Waals parameters were used"* and cites an earlier modified Lennard-Jones potential reference).
> 
> 

---

## Dependencies

* `numpy`
* `jax` (`jax.numpy`, `jax.jit`, `jax.vmap`, `jax.lax.scan`, `jax.random`, `jax.ops`)
* `MDAnalysis` (`MDAnalysis`, `MDAnalysis.topology.guessers.guess_atom_element`)

---

## Van der Waals Parameter Table

`VDW_ELEMENT_PARAMS` maps element symbols to `(epsilon, sigma)` tuples, where:

* `epsilon` is in `kcal/mol` (well depth), and
* `sigma` is in `Angstroms` (collision diameter).

| Key | `epsilon` (`kcal/mol`) | `sigma` (`Angstroms`) | Notes |
| --- | --- | --- | --- |
| `"C"` | `0.1094` | `3.3997` | Representative AMBER/GAFF `c3` carbon-like default |
| `"N"` | `0.1700` | `3.2500` | Representative nitrogen default |
| `"O"` | `0.2100` | `2.9599` | Representative oxygen default |
| `"S"` | `0.2500` | `3.5636` | Representative sulfur default |
| `"P"` | `0.2000` | `3.7418` | Representative phosphorus default |
| `"SE"` | `0.2500` | `3.7418` | Selenomethionine selenium default |
| `"H"` | `0.0157` | `1.4870` | Representative hydrogen default |
| `DEFAULT_VDW_PARAMS` | `0.1200` | `3.4000` | Fallback for unmapped elements |

---

## Data Structures & Functions

### `ResidueKey`

Frozen dataclass storing `chain_id`, `res_seq`, `i_code`, and `res_name`, formatted via `__str__` as `"{chain_id}:{res_seq}{i_code}:{res_name}"`.

### Structure & Trajectory Loading

* **`parse_pdb_for_mcpath(pdb_path, trajectory_path=None, selection="protein", include_hydrogens=True, frame=0, vdw_table=None)`**:
Loads `pdb_path` (and optional `trajectory_path`) into an `MDAnalysis.Universe`, applies `selection` and optional hydrogen filtering, moves to `frame` (supporting negative indices), maps residue indices contiguously to `0 .. N-1`, and returns `(coords, residue_indices, epsilon, sigma, residues, u)`.
* **`compute_trajectory_energy_matrices(u, residue_indices, epsilon, sigma, n_residues, selection="protein", include_hydrogens=True, frames=None, r_cut=5.5)`** *(aliased as `collect_trajectory_energy_stack`)*:
Steps through `frames` (or all frames when `frames is None`) on an existing `MDAnalysis.Universe` and stacks per-frame residue energy matrices into shape `(K, N, N)`.

### JAX Energy, Ensemble & Transition Functions

* **`modified_lj_energy(r, epsilon, sigma, r_cut=5.5, r_min_scale=0.89, e_min_cap=10.0)`**:
JIT-compiled pairwise 12-6 potential that caps repulsion below `r_min_scale * (2**(1/6) * sigma)` at `e_min_cap` and sets energies to `0.0` for `r <= 0` or `r > r_cut`.
* **`build_residue_energy_matrix(coords, residue_indices, epsilon, sigma, n_residues, r_cut=5.5)`**:
JIT-compiled builder that computes pairwise atom distances and energies, zeroes the atom self-interaction diagonal, sums atom-atom terms into an `(N, N)` residue energy matrix via `jax.ops.segment_sum` and `jax.vmap`, and zeroes the residue diagonal.
* **`combine_trajectory_energies(energy_stack, weights=None)`**:
JIT-compiled weighted sum (`jnp.einsum("k,kij->ij", norm_weights, energy_stack)`) over a `(K, N, N)` stack using uniform weights if `weights is None` or normalized non-negative weights otherwise.
* **`boltzmann_frame_weights(frame_energies, temperature=300.0, kb=0.00198720425864083)`**:
JIT-compiled normalized Boltzmann weights from a 1D `(K,)` array of per-frame energies.
* **`energy_to_transition_matrix(res_energy, temperature=300.0, kb=0.00198720425864083)`**:
JIT-compiled row-normalized Boltzmann transition probability matrix with zero diagonal (`P[i, i] = 0`).
* **`ensemble_transition_matrix_from_universe(...)`**:
Orchestrates trajectory energy collection, optional frame or Boltzmann weighting, energy combination, and transition matrix construction, returning `(energy_stack, combined_energy, prob_matrix)`.

---

## `JAXMCPathAnalyzer`

* **`__init__(prob_matrix, residues=None, seed=42)`**: Validates a square `(N, N)` transition matrix and initializes a JAX PRNG key.
* **`label(idx)`**: Formats residue index `idx` with its `ResidueKey` if available.
* **`path_probability(path)`**: Computes the product of step transition probabilities along a sequence of residue indices.
* **`sample_from_residue(start_res, n_steps, n_samples=5000)`**: Mode (A) fixed-length walk sampling from `start_res`, returning `(path, count, path_prob)` sorted by descending count and probability.
* **`sample_between_residues(start_res, end_res, max_steps=12, n_samples=10000)`**: Mode (B) walk sampling from `start_res` up to `max_steps`, retaining trajectories that reach `end_res` (truncated at first arrival) and sorting by descending count and probability.
* **`compute_walk_centrality(walk_length=2000, n_walks=50)`**: Mode (C) multi-walk simulation estimating normalized visit frequency (`betweenness_proxy`), inverse mean return interval (`closeness_proxy`), and `mean_return_steps` per residue.

---

## Example Usage


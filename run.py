from pathlib import Path
import jax.numpy as jnp
from jaxmcprotpath import (
    JAXMCPathAnalyzer,
    ensemble_transition_matrix_from_universe,
    parse_pdb_for_mcpath,
)

coords, res_idx, eps, sig, residues, u = parse_pdb_for_mcpath(
    pdb_path=Path("1BE9.pdb"),
    trajectory_path=None,
    selection="protein",
    include_hydrogens=False,
    frame=0,
)

_, _, prob = ensemble_transition_matrix_from_universe(
    u=u,
    residue_indices=res_idx,
    epsilon=eps,
    sigma=sig,
    n_residues=len(residues),
    selection="protein",
    include_hydrogens=False,
    frames=[0],
    temperature=300.0,
)

analyzer = JAXMCPathAnalyzer(prob, residues=residues, seed=42)
centrality = analyzer.compute_walk_centrality(walk_length=2000, n_walks=50)
top_res = sorted(centrality.items(), key=lambda kv: -kv[1]["betweenness_proxy"])[:5]
for idx, scores in top_res:
    print(analyzer.label(idx), scores)
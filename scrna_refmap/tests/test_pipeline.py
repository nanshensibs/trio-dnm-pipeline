"""Command-line smoke test: build a reference and map a query end to end."""
import json
import os

import pandas as pd

from refmap.cli import main
from refmap.simulate import simulate_reference_query


def test_cli_build_map(tmp_path, capsys):
    ref_ad, q, _ = simulate_reference_query(
        n_genes=500, n_ref_batches=2, ref_samples_per_batch=2, cells_per_ref_sample=200,
        n_query_ctrl=3, n_query_dis=3, cells_per_query_sample=200, seed=1)
    for a in (ref_ad, q):
        a.obs = a.obs.astype({c: str for c in a.obs.columns if a.obs[c].dtype == object})
    ref_ad.write_h5ad(tmp_path / "ref.h5ad")
    q.write_h5ad(tmp_path / "q.h5ad")
    main(["build", "--h5ad", str(tmp_path / "ref.h5ad"), "--labels", "cell_type_l1,cell_type_l2",
          "--batch-key", "batch", "--sample-key", "sample", "--n-hvg", "500", "--n-pcs", "20",
          "--out", str(tmp_path / "ref")])
    built = json.loads(capsys.readouterr().out)
    assert built["self_mapping_accuracy"]["cell_type_l1"] > 0.95
    out = tmp_path / "res"
    main(["map", "--reference", str(tmp_path / "ref"), "--query", str(tmp_path / "q.h5ad"),
          "--out", str(out), "--sample-key", "sample", "--condition-key", "condition",
          "--case", "disease", "--control", "control", "--novel-key", "is_novel",
          "--no-umap", "--no-augur"])
    for f in ("mapping.tsv", "query_clusters.tsv", "composition.tsv", "milo_nhoods.tsv",
              "pseudobulk_de.tsv", "metrics.json", "query_mapped.h5ad", "report.html"):
        assert os.path.exists(out / f), f
    m = pd.read_csv(out / "mapping.tsv", sep="\t", index_col=0)
    assert {"cell_type_l2_pred", "cell_type_l2_uncertainty", "map_ood"} <= set(m.columns)
    comp = pd.read_csv(out / "composition.tsv", sep="\t").set_index("cell_type")
    assert comp.loc["CD14 Mono", "log2FC"] > 0

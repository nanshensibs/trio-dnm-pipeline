"""Open-source atlasing: versioned registries, reviewed updates, reference harmonisation.

Implements the perspective's section "Path toward machine-learning-based open-source
atlasing": references are released and improved like open-source software --

* ``ReferenceRegistry`` -- a directory of versioned references
  (``<root>/<name>/<semver>/``, written with ``Reference.save``) with a JSON index of
  content hashes, parent versions and changelogs (the Git-like history the paper asks
  for, so every mapping result can cite an exact, verifiable reference version);
* ``propose_update`` -- the "pull request" for a reference: new labelled data are
  first *mapped* onto the current version (agreement of submitted vs. transferred
  labels, labels the reference lacks, novel query clusters), a candidate version is
  built, semantic-versioned and reviewed (cells added per label, label stability of the
  existing cells, self-mapping accuracy change); ``accept_update`` publishes it.
  Without the old counts the candidate keeps the frozen transformation and only
  appends projected cells before re-integration -- the scArches idea (ref. 3) of
  extending a reference without access to the original raw data;
* ``harmonize_references`` -- cross-reference label harmonisation by reciprocal
  mapping (e.g. HCA vs. LungMAP lung atlases), producing 1:1, split (1:n), merge (n:1)
  and unmatched relations and a merged label tree in the spirit of the "reference cell
  tree" of Domcke & Shendure (ref. 113).

Semantic versions: *patch* = re-annotation only; *minor* = new cells and/or labels
with an unchanged transformation definition; *major* = the transformation changed
(gene set, number of components or transform type), so latent coordinates from
earlier versions are no longer comparable.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import re
import warnings
from dataclasses import dataclass, field

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

from .graph import knn
from .harmony import compress_reference, run_harmony, symphony_map
from .mapping import MappingResult, map_query
from .reference import Reference, _jsonable, build_reference, calibrate_reference
from .transfer import UNKNOWN, knn_weights, transfer_hierarchical

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(v: str) -> tuple[int, int, int]:
    m = _SEMVER.match(str(v))
    if not m:
        raise ValueError(f"not a semantic version (MAJOR.MINOR.PATCH): {v!r}")
    return tuple(int(x) for x in m.groups())


def bump_version(v: str, level: str) -> str:
    major, minor, patch = parse_version(v)
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level {level!r}")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ registry


class ReferenceRegistry:
    """Directory of versioned references with a JSON index (``index.json``)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, "index.json")

    def _read(self) -> dict:
        if not os.path.exists(self.index_path):
            return {"format": "refmap-registry", "references": {}}
        with open(self.index_path) as fh:
            return json.load(fh)

    def _write(self, idx: dict) -> None:
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(_jsonable(idx), fh, indent=2)
        os.replace(tmp, self.index_path)   # atomic

    def path(self, name: str, version: str) -> str:
        return os.path.join(self.root, name, version)

    def names(self) -> list[str]:
        return sorted(self._read()["references"])

    def versions(self, name: str) -> list[str]:
        return sorted(self._read()["references"].get(name, {}), key=parse_version)

    def latest(self, name: str) -> str:
        v = self.versions(name)
        if not v:
            raise KeyError(f"no published versions of {name!r}")
        return v[-1]

    def entry(self, name: str, version: str | None = None) -> dict:
        version = version or self.latest(name)
        try:
            return dict(self._read()["references"][name][version])
        except KeyError:
            raise KeyError(f"{name} {version} is not in the registry") from None

    def history(self, name: str) -> pd.DataFrame:
        rows = [dict(version=v, **{k: e.get(k) for k in
                                   ("parent_version", "content_sha256", "n_cells", "published")},
                     change=(e.get("changelog") or [""])[-1])
                for v in self.versions(name) for e in [self.entry(name, v)]]
        return pd.DataFrame(rows)

    def publish(self, ref: Reference, *, overwrite: bool = False,
                extra: dict | None = None) -> str:
        """Save ``ref`` under ``<root>/<name>/<version>/`` and index it. Re-publishing
        identical content is a no-op; different content under an existing version
        raises (versions are immutable) unless ``overwrite``."""
        parse_version(ref.version)
        h = ref.content_hash()
        idx = self._read()
        entries = idx["references"].setdefault(ref.name, {})
        if ref.version in entries and not overwrite:
            if entries[ref.version]["content_sha256"] == h:
                return self.path(ref.name, ref.version)
            raise ValueError(f"{ref.name} {ref.version} already published with different "
                             "content; bump the version")
        parent = ref.provenance.get("parent_version")
        if parent is not None:
            if parent not in entries:
                warnings.warn(f"parent version {parent} of {ref.name} is not in the "
                              "registry", stacklevel=2)
            elif parse_version(ref.version) <= parse_version(parent):
                raise ValueError(f"version {ref.version} is not newer than parent {parent}")
        path = ref.save(self.path(ref.name, ref.version))
        entries[ref.version] = dict(
            content_sha256=h, parent_version=parent,
            changelog=list(ref.provenance.get("changelog", [])),
            datasets=list(ref.provenance.get("datasets", [])),
            created=ref.provenance.get("created"), published=_now(),
            n_cells=ref.n_cells, n_genes=int(len(ref.genes)), n_dims=ref.n_dims,
            label_keys=list(ref.label_keys),
            path=os.path.relpath(path, self.root), **(extra or {}))
        self._write(idx)
        return path

    def load(self, name: str, version: str | None = None, verify: bool = True) -> Reference:
        version = version or self.latest(name)
        e = self.entry(name, version)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # hash is checked against the index below
            ref = Reference.load(os.path.join(self.root, e["path"]))
        if verify and ref.content_hash() != e["content_sha256"]:
            raise ValueError(f"content hash of {name} {version} does not match the registry")
        return ref


# ---------------------------------------------------------- update proposals


@dataclass
class UpdateProposal:
    candidate: Reference
    report: dict
    bump: str
    mapping: MappingResult | None = None     # submission mapped onto the parent

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(_jsonable(self.report), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s


def _check_hierarchy(obs: pd.DataFrame, label_keys: list) -> None:
    for parent, child in zip(label_keys[:-1], label_keys[1:]):
        n = obs.groupby(child, observed=True)[parent].nunique()
        if (n > 1).any():
            raise ValueError(f"labels in '{child}' have more than one '{parent}' parent: "
                             f"{list(n[n > 1].index)}")


def _map_cells_loo(ref: Reference, Z: np.ndarray, batch, self_rows: np.ndarray,
                   k: int) -> pd.DataFrame:
    """Map cells that are themselves part of ``ref`` (projected latent ``Z``) through
    ``ref`` with Symphony + kNN, excluding each cell's own entry from its neighbours."""
    Zc, _ = symphony_map(Z, ref.comp, batch)
    dist, idx = knn(ref.Z_corr, Zc, k=k + 1)
    keep = idx != self_rows[:, None]
    keep[keep.all(axis=1), -1] = False         # self not found: drop the farthest
    dist = dist[keep].reshape(len(Z), k)
    idx = idx[keep].reshape(len(Z), k)
    return transfer_hierarchical(ref.obs, ref.label_keys, idx, knn_weights(dist),
                                 unknown_threshold=1.0)


def _submission_report(ref: Reference, m: MappingResult, sub: pd.DataFrame) -> dict:
    """(a) agreement of submitted vs. transferred labels, unseen labels, novel clusters."""
    out = dict(n_cells=int(len(sub)), frac_missing_genes=m.frac_missing_genes, levels={})
    novel_cl = set(m.cluster_summary.loc[m.cluster_summary["novel"], "cluster"])
    in_novel = np.isin(m.clusters, list(novel_cl))
    ood = m.scores["ood"].to_numpy()
    for key in ref.label_keys:
        s = sub[key].to_numpy()
        t = m.labels[f"{key}_pred"].astype(str).to_numpy()
        known = np.isin(s, ref.labels(key))
        per = (pd.DataFrame({"s": s[known], "ok": s[known] == t[known]})
               .groupby("s")["ok"].mean().round(4).to_dict())
        new = {}
        for lab in sorted(set(s[~known])):
            mm = s == lab
            new[lab] = dict(n_cells=int(mm.sum()),
                            frac_in_novel_clusters=float(in_novel[mm].mean()),
                            frac_ood=float(ood[mm].mean()),
                            transferred_as=pd.Series(t[mm]).value_counts(normalize=True)
                            .head(3).round(3).to_dict())
        out["levels"][key] = dict(
            agreement=float((s[known] == t[known]).mean()) if known.any() else None,
            agreement_per_label=per, new_labels=new,
            confusion=pd.crosstab(pd.Series(s, name="submitted"),
                                  pd.Series(t, name="transferred")).to_dict(orient="index"))
    fine = sub[ref.label_keys[-1]].to_numpy()
    out["novel_clusters"] = [
        dict(cluster=int(c), n_cells=int((m.clusters == c).sum()),
             submitted=pd.Series(fine[m.clusters == c]).value_counts(normalize=True)
             .head(3).round(3).to_dict())
        for c in sorted(novel_cl)]
    return out


def propose_update(ref: Reference, new_adata=None, label_keys=None, *,
                   batch_key: str | None = None, sample_key: str | None = None,
                   dataset: str | None = None, ref_adata=None, relabel: pd.DataFrame | None = None,
                   refresh_hvgs: bool = False, n_hvg: int | None = None,
                   n_pcs: int | None = None, transform: str | None = None,
                   message: str | None = None, max_accuracy_drop: float = 0.05,
                   map_kwargs: dict | None = None, seed: int = 0) -> UpdateProposal:
    """Build and review a candidate new version of ``ref``.

    ``new_adata``: raw counts with annotations in ``label_keys`` (columns aligned with
    ``ref.label_keys``, default the same names) and batches in ``batch_key``.
    ``relabel``: DataFrame indexed by reference cell names with corrected labels for
    some ``ref.label_keys`` columns (re-annotation). ``ref_adata``: the reference's raw
    counts; when given, the candidate is rebuilt from merged counts with
    ``build_reference`` (keeping ``ref.genes`` unless ``refresh_hvgs``; ``n_pcs`` /
    ``transform`` may change), otherwise the transformation is frozen and new cells are
    projected and appended before re-running Harmony.
    """
    if new_adata is None and relabel is None:
        raise ValueError("nothing to update: give new_adata and/or relabel")
    keys = list(ref.label_keys)
    label_keys = list(label_keys) if label_keys is not None else keys
    if len(label_keys) != len(keys):
        raise ValueError("label_keys must align one-to-one with ref.label_keys")
    rebuild = ref_adata is not None
    if not rebuild and (refresh_hvgs or n_pcs or transform or n_hvg):
        raise ValueError("changing the transformation needs ref_adata (raw counts)")
    k = int(ref.params.get("k_calibration", 30))
    map_kwargs = dict(map_kwargs or {})
    bkey = ref.batch_key or "dataset"

    # ---- old cells (with re-annotation) ------------------------------------
    old_obs = ref.obs.copy()
    if bkey not in old_obs:
        old_obs[bkey] = "reference"
    n_relabelled = {}
    if relabel is not None:
        relabel = relabel.reindex(columns=[c for c in relabel.columns if c in keys])
        for c in relabel.columns:
            v = relabel[c].dropna().astype(str)
            v = v[v.index.isin(old_obs.index)]
            n_relabelled[c] = int((old_obs.loc[v.index, c] != v).sum())
            old_obs.loc[v.index, c] = v
    if "added_in" not in old_obs:
        old_obs["added_in"] = ref.version

    # ---- (a) map the submission onto the current version --------------------
    mapping, submission, new_obs = None, None, None
    if new_adata is not None:
        mapping = map_query(ref, new_adata, batch_key=batch_key, seed=seed, **map_kwargs)
        new_obs = pd.DataFrame({k_ref: new_adata.obs[k_new].astype(str).to_numpy()
                                for k_ref, k_new in zip(keys, label_keys)},
                               index=np.asarray(new_adata.obs_names).astype(str))
        submission = _submission_report(ref, mapping, new_obs)
        dataset = dataset or f"submission_{len(ref.provenance.get('datasets', [])) + 1}"
        new_obs[bkey] = (new_adata.obs[batch_key].astype(str).to_numpy() if batch_key
                         else dataset)
        if ref.sample_key:
            new_obs[ref.sample_key] = (new_adata.obs[sample_key].astype(str).to_numpy()
                                       if sample_key else new_obs[bkey])
        if new_obs.index.isin(old_obs.index).any():
            new_obs.index = [f"{dataset}:{i}" for i in new_obs.index]

    # ---- (b) candidate ---------------------------------------------------------
    obs = old_obs if new_obs is None else pd.concat([old_obs, new_obs])
    _check_hierarchy(obs, keys)
    params = ref.params
    new_n_pcs = n_pcs or params.get("n_pcs")
    new_transform = transform or params.get("transform", "pca")
    if rebuild:
        old_ad = ref_adata[np.asarray(ref.obs.index)].copy()
        old_ad.obs = old_obs.loc[old_ad.obs_names].copy()
        parts = [old_ad]
        if new_adata is not None:
            nad = new_adata.copy()
            nad.obs = new_obs.copy()
            nad.obs_names = new_obs.index
            parts.append(nad)
        merged = ad.concat(parts, join="inner", merge="same") if len(parts) > 1 else parts[0]
        merged.obs = obs.loc[merged.obs_names].copy()
        hvg = None if refresh_hvgs else [g for g in ref.genes if g in set(merged.var_names)]
        cand = build_reference(
            merged, keys, batch_key=bkey, sample_key=ref.sample_key, name=ref.name,
            version=ref.version, n_hvg=n_hvg or params.get("n_hvg", 2000), n_pcs=new_n_pcs,
            transform=new_transform, hvg_genes=hvg, seed=seed,
            harmony_kwargs={k_: v for k_, v in params.get("harmony", {}).items()
                            if k_ not in ("sigma", "seed")},
            sigma=params.get("harmony", {}).get("sigma", 0.1), k_calibration=k,
            calibrate=True)
    else:
        if new_adata is not None:
            Zn, _ = ref.project(new_adata.X, new_adata.var_names)
            Z = np.vstack([ref.Z, Zn])
            hk = dict(params.get("harmony", {}))
            h = run_harmony(Z, obs[bkey].astype(str).to_numpy(), **hk)
            comp, _ = compress_reference(h.Z_corr, h.Y, ref.comp.sigma, ref.comp.lamb)
            cont = {}
            for c, v in ref.continuous.items():
                if c in new_adata.obsm:
                    cont[c] = np.vstack([v, np.asarray(new_adata.obsm[c], float)])
            new_params = dict(params, harmony_converged=bool(h.converged),
                              n_centroids=int(len(comp.Nr)))
            cand = dataclasses.replace(ref, Z=Z, Z_corr=h.Z_corr, comp=comp,
                                       continuous=cont,
                                       continuous_names={c: ref.continuous_names[c]
                                                         for c in cont},
                                       params=new_params, batch_key=bkey)
        else:
            cand = dataclasses.replace(ref, params=dict(params), batch_key=bkey)
        cand.obs = obs.copy()
        cand.calibration = calibrate_reference(cand, None, k=k, seed=seed)

    # ---- semantic version + provenance --------------------------------------
    transform_changed = (not np.array_equal(cand.genes.astype(str), ref.genes.astype(str))
                         or int(cand.params.get("n_pcs")) != int(params.get("n_pcs"))
                         or cand.params.get("transform") != params.get("transform"))
    new_labels = {key: sorted(set(obs[key]) - set(ref.labels(key))) for key in keys}
    if transform_changed:
        bump = "major"
    elif new_obs is not None or any(new_labels.values()):
        bump = "minor"
    else:
        bump = "patch"
    version = bump_version(ref.version, bump)
    cand.version = version
    if new_obs is not None:
        cand.obs.loc[new_obs.index, "added_in"] = version
    auto = []
    if new_obs is not None:
        auto.append(f"added {len(new_obs)} cells from {dataset}")
    if any(new_labels.values()):
        auto.append("new labels " + ", ".join(sorted({l for v in new_labels.values() for l in v})))
    if n_relabelled:
        auto.append(f"re-annotated {max(n_relabelled.values())} cells")
    if transform_changed:
        auto.append("transformation changed")
    datasets = list(ref.provenance.get("datasets", []))
    if new_obs is not None:
        datasets += [d for d in pd.unique(new_obs[bkey]) if d not in datasets]
    cand.provenance = dict(
        ref.provenance, created=_now(), parent_version=ref.version,
        parent_sha256=ref.content_hash(), datasets=datasets,
        changelog=list(ref.provenance.get("changelog", []))
        + [f"{version}: {message or '; '.join(auto)}"])

    # ---- (c) review report ----------------------------------------------------
    n_old = ref.n_cells
    rows = np.arange(n_old)
    old_bat = None if ref.batch_key is None else ref.obs[ref.batch_key].to_numpy()
    before = _map_cells_loo(ref, ref.Z, old_bat, rows, k)
    after = _map_cells_loo(cand, cand.Z[:n_old], cand.obs[bkey].to_numpy()[:n_old], rows, k)
    stability = {}
    for key in keys:
        y = ref.obs[key].astype(str).to_numpy()
        b = before[f"{key}_pred"].to_numpy() == y
        a = after[f"{key}_pred"].to_numpy() == y
        per = pd.DataFrame({"l": y, "b": b, "a": a}).groupby("l")[["b", "a"]].mean()
        per["change"] = per["a"] - per["b"]
        stability[key] = dict(
            old_version=float(b.mean()), new_version=float(a.mean()),
            change=float(a.mean() - b.mean()),
            most_changed=per.sort_values("change").head(5).round(4).to_dict(orient="index"),
            reassigned_to=pd.Series(after[f"{key}_pred"].to_numpy()[~a & b])
            .value_counts().head(5).to_dict())
    acc_old = ref.calibration.get("self_mapping_accuracy", {})
    acc_new = cand.calibration.get("self_mapping_accuracy", {})
    self_map = {key: dict(old=acc_old.get(key), new=acc_new.get(key),
                          change=(acc_new[key] - acc_old[key]
                                  if key in acc_old and key in acc_new else None))
                for key in keys}
    checks = dict(
        self_mapping_ok=all(v["change"] is None or v["change"] >= -max_accuracy_drop
                            for v in self_map.values()),
        old_label_stability_ok=all(v["change"] >= -max_accuracy_drop
                                   for v in stability.values()),
        hierarchy_ok=True)
    report = dict(
        reference=ref.name, parent_version=ref.version, candidate_version=version,
        bump=bump, mode="rebuild" if rebuild else "extend (frozen transformation)",
        transform_changed=transform_changed, dataset=dataset if new_obs is not None else None,
        submission=submission,
        cells_added_per_label={key: (new_obs[key].value_counts().to_dict()
                                     if new_obs is not None else {}) for key in keys},
        new_labels=new_labels, relabelled_cells=n_relabelled,
        label_counts={key: dict(old=ref.obs[key].value_counts().to_dict(),
                                new=cand.obs[key].value_counts().to_dict()) for key in keys},
        old_cell_label_agreement=stability, self_mapping_accuracy=self_map,
        checks=checks, passed=all(checks.values()),
        parent_sha256=ref.content_hash(), candidate_sha256=cand.content_hash(),
        changelog_entry=cand.provenance["changelog"][-1])
    return UpdateProposal(candidate=cand, report=_jsonable(report), bump=bump,
                          mapping=mapping)


def accept_update(registry: ReferenceRegistry, candidate, *, force: bool = False) -> str:
    """Merge the "pull request": publish the candidate (a ``Reference`` or an
    ``UpdateProposal``, whose review must have passed unless ``force``) and store its
    review report next to it."""
    report = None
    if isinstance(candidate, UpdateProposal):
        if not candidate.report.get("passed", False) and not force:
            raise ValueError(f"review checks failed: {candidate.report.get('checks')}; "
                             "pass force=True to publish anyway")
        report, candidate = candidate.report, candidate.candidate
    path = registry.publish(candidate, extra=dict(review_passed=report.get("passed"))
                            if report else None)
    if report is not None:
        with open(os.path.join(path, "review.json"), "w") as fh:
            json.dump(_jsonable(report), fh, indent=2)
    return path


# ------------------------------------------------------------- harmonisation


@dataclass
class HarmonizationResult:
    table: pd.DataFrame             # one row per harmonised label group
    edges: pd.DataFrame             # supported (label_A, label_B) pairs with fractions
    contingency_ab: pd.DataFrame    # A cells: A label x transferred B label (counts)
    contingency_ba: pd.DataFrame    # B cells: B label x transferred A label (counts)
    tree: dict = field(default_factory=dict)


def _transferred(ref_to: Reference, adata, level_to: str, batch_key, map_kwargs) -> np.ndarray:
    m = map_query(ref_to, adata, batch_key=batch_key, **map_kwargs)
    return m.labels[f"{level_to}_pred"].astype(str).to_numpy()


def harmonize_references(refA: Reference, refB: Reference, adataA, adataB,
                         level: str | None = None, *, level_a: str | None = None,
                         level_b: str | None = None, batch_key_a: str | None = None,
                         batch_key_b: str | None = None, min_fraction: float = 0.1,
                         majority: float = 0.5, map_kwargs: dict | None = None
                         ) -> HarmonizationResult:
    """Reciprocal mapping of two references' cells and label-relation inference.

    A's cells (``adataA``, annotated in ``level_a``) are mapped onto B and B's onto A,
    giving P(b | a) and P(a | b). A pair (a, b) is linked when one direction is a
    majority (>= ``majority``) and the other carries at least ``min_fraction`` -- so a
    coarse A label whose cells spread over several B labels, each of which maps back
    to it, is a *split*. Connected components of the bipartite link graph are
    classified as ``equivalent`` (1:1), ``split`` (1 A : n B), ``merge`` (n A : 1 B),
    ``complex`` (n:m), ``A_only`` / ``B_only`` (unmatched, candidate novel states).
    The tree nests harmonised groups under A's (else B's) parent labels when the level
    has a parent, with the finer labels of splits/merges as children.
    """
    level_a = level_a or level or refA.label_keys[-1]
    level_b = level_b or level or refB.label_keys[-1]
    mk = dict(map_kwargs or {})
    ba = batch_key_a or (refA.batch_key if refA.batch_key in adataA.obs else None)
    bb = batch_key_b or (refB.batch_key if refB.batch_key in adataB.obs else None)
    la = adataA.obs[level_a].astype(str).to_numpy()
    lb = adataB.obs[level_b].astype(str).to_numpy()
    a_on_b = _transferred(refB, adataA, level_b, ba, mk)
    b_on_a = _transferred(refA, adataB, level_a, bb, mk)
    C_ab = pd.crosstab(pd.Series(la, name=f"A:{level_a}"), pd.Series(a_on_b, name=f"B:{level_b}"))
    C_ba = pd.crosstab(pd.Series(lb, name=f"B:{level_b}"), pd.Series(b_on_a, name=f"A:{level_a}"))
    P_ab = C_ab.div(C_ab.sum(axis=1), axis=0)        # P(b | a)
    P_ba = C_ba.div(C_ba.sum(axis=1), axis=0)        # P(a | b)
    A_labels, B_labels = list(C_ab.index), list(C_ba.index)

    def p(P, r, c):
        return float(P.at[r, c]) if r in P.index and c in P.columns else 0.0

    G = nx.Graph()
    G.add_nodes_from(("A", a) for a in A_labels)
    G.add_nodes_from(("B", b) for b in B_labels)
    edges = []
    for a in A_labels:
        for b in B_labels:
            if UNKNOWN in (a, b):
                continue
            pab, pba = p(P_ab, a, b), p(P_ba, b, a)
            if (pab >= min_fraction and pba >= majority) or (pba >= min_fraction and
                                                             pab >= majority):
                G.add_edge(("A", a), ("B", b))
                edges.append(dict(label_A=a, label_B=b, p_b_given_a=round(pab, 4),
                                  p_a_given_b=round(pba, 4)))
    edges = pd.DataFrame(edges, columns=["label_A", "label_B", "p_b_given_a", "p_a_given_b"])

    rows = []
    for comp in nx.connected_components(G):
        A = sorted(l for s, l in comp if s == "A")
        B = sorted(l for s, l in comp if s == "B")
        if len(A) == 1 and len(B) == 1:
            a, b = A[0], B[0]
            recip = (P_ab.loc[a].idxmax() == b) and (P_ba.loc[b].idxmax() == a)
            rel, name, children = ("equivalent" if recip else "equivalent_weak",
                                   a if a == b else f"{a} | {b}", [])
        elif len(A) == 1 and len(B) > 1:
            rel, name, children = "split", A[0], B
        elif len(A) > 1 and len(B) == 1:
            rel, name, children = "merge", B[0], A
        elif len(B) == 0:
            rel, name, children = "A_only", A[0], []
        elif len(A) == 0:
            rel, name, children = "B_only", B[0], []
        else:
            rel, name, children = "complex", " + ".join(A), B
        rows.append(dict(harmonized_label=name, relation=rel, labels_A=A, labels_B=B,
                         children=children,
                         n_cells_A=int(C_ab.loc[A].to_numpy().sum()) if A else 0,
                         n_cells_B=int(C_ba.loc[B].to_numpy().sum()) if B else 0))
    table = (pd.DataFrame(rows).sort_values(["relation", "harmonized_label"])
             .reset_index(drop=True))

    parent_a = refA.hierarchy().get(level_a, {})
    parent_b = refB.hierarchy().get(level_b, {})
    root: dict = {}
    for r in table.itertuples():
        par = (parent_a.get(r.labels_A[0]) if r.labels_A else None) or \
              (parent_b.get(r.labels_B[0]) if r.labels_B else None)
        node = {c: {} for c in r.children}
        (root.setdefault(par, {}) if par else root)[r.harmonized_label] = node
    return HarmonizationResult(table=table, edges=edges, contingency_ab=C_ab,
                               contingency_ba=C_ba, tree={"root": root})


__all__ = ["parse_version", "bump_version", "ReferenceRegistry", "UpdateProposal",
           "propose_update", "accept_update", "HarmonizationResult", "harmonize_references"]

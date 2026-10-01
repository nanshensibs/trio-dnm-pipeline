"""Regression tests for the Nextflow-side review fixes: the bin/ scripts and
static checks of the workflow files (Nextflow itself is optional: the stub run
at the end is skipped unless `nextflow` is on PATH)."""
import gzip
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
NF_FILES = [os.path.join(ROOT, "main.nf")] + sorted(
    os.path.join(ROOT, "modules", f) for f in os.listdir(os.path.join(ROOT, "modules")) if f.endswith(".nf"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BIN, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(path):
    with open(path) as fh:
        return fh.read()


def _processes():
    """{name: body} of every `process NAME { ... }` block (closing brace at column 0),
    without full-line // comments."""
    out = {}
    for path in NF_FILES:
        for m in re.finditer(r"^process\s+(\w+)\s*\{\n(.*?)^\}", _text(path), re.S | re.M):
            out[m.group(1)] = "".join(ln for ln in m.group(2).splitlines(True) if not ln.lstrip().startswith("//"))
    return out


# --------------------------------------------------------------------------- #
# bin/vcf2mfbed.py (NF-11): SNVs only, indels counted on stderr

M2_VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tKID
chr1\t100\t.\tC\tT\t.\tPASS\t.\tGT:AF\t0/1:0.10
chr1\t200\t.\tCA\tC\t.\tPASS\t.\tGT:AF\t0/1:0.12
chr1\t300\t.\tG\tGTT\t.\tPASS\t.\tGT:AF\t0/1:0.20
chr1\t400\t.\tA\tG\t.\tPASS\t.\tGT:AF\t0/1:0.48
chr1\t500\t.\tA\tG,T\t.\tPASS\t.\tGT:AF\t0/1:0.1,0.1
chr1\t600\t.\tA\tC\t.\tweak_evidence\t.\tGT:AF\t0/1:0.1
chr1\t700\t.\tA\t*\t.\tPASS\t.\tGT:AF\t0/1:0.1
chr1\t800\t.\tg\tc\t.\t.\t.\tGT:AF\t0/1:0.05
"""


@pytest.mark.parametrize("gz", [False, True])
def test_vcf2mfbed_keeps_pass_biallelic_snvs_only(tmp_path, capsys, gz):
    mod = _load("vcf2mfbed")
    p = tmp_path / ("m2.vcf.gz" if gz else "m2.vcf")
    if gz:
        with gzip.open(p, "wt") as fh:
            fh.write(M2_VCF)
    else:
        p.write_text(M2_VCF)
    mod.main([str(p), "KID", "--min-vaf", "0.03", "--max-vaf", "0.40"])
    out, err = capsys.readouterr()
    rows = [line.split("\t") for line in out.strip().split("\n")]
    assert rows == [["chr1", "99", "100", "C", "T", "KID"], ["chr1", "799", "800", "g", "c", "KID"]]
    assert "2 PASS biallelic indel(s) skipped" in err and "2 SNV site(s)" in err


def test_vcf2mfbed_unknown_sample_is_a_clear_error(tmp_path):
    mod = _load("vcf2mfbed")
    p = tmp_path / "m2.vcf"
    p.write_text(M2_VCF)
    with pytest.raises(SystemExit) as e:
        mod.main([str(p), "OTHER"])
    assert "OTHER" in str(e.value) and "KID" in str(e.value)


# --------------------------------------------------------------------------- #
# bin/trio-dnm (NF-01): importable from a staged pylib/ when the project
# directory is not visible (container layout), with a clear error otherwise.

def _run_launcher(launcher, cwd, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run([sys.executable, launcher, "--version"], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)


def test_launcher_runs_from_project_bin(tmp_path):
    r = _run_launcher(os.path.join(BIN, "trio-dnm"), str(tmp_path))
    assert r.returncode == 0 and r.stdout.strip() == "2.0.0"


def test_launcher_uses_staged_package_without_project_dir(tmp_path):
    # Only bin/ is mounted; the task directory holds pylib/trio_dnm (as staged by Nextflow).
    mounted_bin = tmp_path / "mounted_bin"
    mounted_bin.mkdir()
    launcher = mounted_bin / "trio-dnm"
    shutil.copy(os.path.join(BIN, "trio-dnm"), launcher)
    task = tmp_path / "task"
    (task / "pylib").mkdir(parents=True)
    os.symlink(os.path.join(ROOT, "trio_dnm"), task / "pylib" / "trio_dnm")
    r = _run_launcher(str(launcher), str(task), pythonpath=str(task / "pylib"))
    assert r.returncode == 0 and r.stdout.strip() == "2.0.0", r.stderr
    r = _run_launcher(str(launcher), str(task))
    assert r.returncode != 0 and "cannot import the trio_dnm package" in r.stderr
    assert "Traceback" not in r.stderr


# --------------------------------------------------------------------------- #
# Static checks of the workflow files

RESOURCE_PARAMS = [
    "intervals", "targets", "somalier_sites", "verifybamid_svd_prefix", "vcfanno_toml", "exclude_beds",
    "pon", "recurrence", "dnm_config", "vep_cache", "vep_plugin_dir", "gene_table", "cosmic_sbs",
    "validated", "germline_resource", "m2_pon", "common_biallelic", "ch_genes_bed", "mf_umap_bigwig",
    "mf_model", "known_sites", "bwamem2_index",
]


def test_every_process_has_a_stub():
    procs = _processes()
    assert len(procs) >= 40
    missing = [n for n, body in procs.items() if not re.search(r"^\s*stub:", body, re.M)]
    assert not missing, missing


def test_resources_are_staged_inputs_not_host_paths():
    # NF-02: a host path interpolated into a command is not mounted into the container.
    for name, body in _processes().items():
        for p in RESOURCE_PARAMS:
            assert f"params.{p}" not in body, f"{name} uses params.{p} directly"


def test_trio_dnm_processes_run_in_the_python_container_with_the_staged_package():
    # NF-01 / item 13: trio-dnm and the bin/*.py helpers run in params.containers.trio_dnm
    # (python3 >= 3.10), with the package staged as pylib/trio_dnm.
    users = 0
    for name, body in _processes().items():
        script = "".join(ln for ln in body.split("stub:")[0].splitlines(True) if not ln.lstrip().startswith("#"))
        if re.search(r"\btrio-dnm\b", script):
            assert "path(pkg, stageAs: 'pylib/trio_dnm')" in body and "trioDnmEnv()" in script, name
        if re.search(r"\btrio-dnm\b|dng2tsv\.py|vcf2mfbed\.py", script):
            assert "container params.containers.trio_dnm" in body, name
            users += 1
    assert users >= 7   # 5 trio-dnm commands + PROVENANCE + MOSAIC_SITES_BED


def test_tool_invocations_fixed():
    procs = _processes()
    assert re.search(r"--bam-pairs \$\{kid\}:\$\{bam\}", procs["UNFAZED"])                       # NF-03
    assert "--vcf ${cand_vcf}" in procs["DENOVOGEAR"] and "--bcf" not in procs["DENOVOGEAR"]     # NF-04
    assert "-a AD,ADF,ADR,DP" in procs["MPILEUP_CANDIDATES"]                                       # F10
    assert "configManta.py" in procs["MANTA"] and "configManta.py" not in procs["STRELKA2"]        # NF-05
    assert "--indelCandidates ${indels}" in procs["STRELKA2"]
    assert "cnv/*.cns" not in procs["CNVKIT"] and "! -name '*.call.cns'" in procs["CNVKIT"]       # NF-06
    assert '$i == "prediction"' in procs["PARENTAL_ABSENCE"]                                       # NF-07
    for line in re.findall(r"gatk GetPileupSummaries[^\n]*", procs["MUTECT2_PAIRED"]):            # NF-08
        assert "-R ${fasta}" in line
    for name in ("MPILEUP_CANDIDATES", "PARENTAL_ABSENCE"):                                        # NF-09
        mp = re.findall(r"bcftools mpileup[^\n]*", procs[name])
        assert mp and all(" -R " in m and " -T " not in m for m in mp)
    assert '"QUAL < 30.0" --filter-name QUAL30' in procs["GENOTYPE_TRIO"].split("snv.f.vcf.gz")[1]  # F18
    assert "-A StrandBiasBySample" in procs["GENOTYPE_TRIO"]                                       # F10
    assert "--extra-vcf ${dt_vcf}" in procs["TRIO_DNM_CALL"] and "--strand-vcf" in procs["TRIO_DNM_CALL"]  # F17
    assert "task.exitStatus == 3 ? 'terminate'" in procs["TRIO_DNM_CALL"]


def test_gnomad_grpmax_field_name():
    # F19: gnomAD v4.1 joint VCFs name the field AF_grpmax_joint.
    for path in [os.path.join(ROOT, "conf", "vcfanno.toml"), os.path.join(ROOT, "modules", "dnm.nf")]:
        text = _text(path)
        assert "AF_joint_grpmax" not in text and "AF_grpmax_joint" in text, path
    assert "sed -i 's/gnomADv4_AF_joint" not in _text(os.path.join(ROOT, "modules", "dnm.nf"))   # F14


def test_retry_scales_resources():
    # NF-12: retried kills get more memory/time.
    cfg = _text(os.path.join(ROOT, "nextflow.config"))
    for label in ("small", "medium", "large"):
        line = re.search(rf"withLabel: {label}\s*\{{([^\n]*)\}}", cfg).group(1)
        assert "memory = {" in line and "time = {" in line and line.count("task.attempt") == 2, line


def test_align_subworkflow_and_reporting_exist():
    # NF-13 / F15 / F25
    procs = _processes()
    for name in ("FASTQC", "FASTP", "BWAMEM2", "MARKDUPLICATES", "BQSR", "MULTIQC", "PROVENANCE"):
        assert name in procs, name
    assert "PU:" in procs["BWAMEM2"] and "--known-sites" in procs["BQSR"]
    main = _text(os.path.join(ROOT, "main.nf"))
    assert "include { ALIGN" in main and "workflow.onComplete" in main and "provenance.json" in main


MF_PREDICTIONS = (
    "id\tconflict_num\ttype\tprediction\thet\tmosaic\trefhom\trepeat\tchromosome\n"
    "s~chr1~100~C~T\t0\tSNP\tmosaic\t0.1\t0.8\t0.05\t0.05\tautosomal\n"
    "s~chr2~200~G~A\t0\tSNP\tmosaic;cautious:AF<0.01\t0.1\t0.8\t0.05\t0.05\tautosomal\n"
    "s~chr3~300~T~C\t0\tSNP\thet\t0.9\t0.05\t0.05\t0\tautosomal\n"
    "s~chrX~400~A~G\t0\tSNP\trefhom\t0.1\t0.1\t0.8\t0\tX\n"
)


@pytest.mark.skipif(not shutil.which("awk"), reason="awk not available")
def test_parental_absence_selects_mosaic_predictions_by_header(tmp_path):
    # NF-07: run the awk program from PARENTAL_ABSENCE on Prediction.R-shaped output.
    body = _processes()["PARENTAL_ABSENCE"]
    prog = re.search(r"awk -F'\\\\t' '(NR == 1 .*?)' \$\{pred\}", body, re.S).group(1)
    prog = prog.replace("\\$", "$").replace("\\\\t", "\\t")
    pred = tmp_path / "pred.txt"
    pred.write_text(MF_PREDICTIONS)
    r = subprocess.run(["awk", "-F\t", prog, str(pred)], capture_output=True, text=True, check=True)
    assert r.stdout.splitlines() == ["chr1\t100\tmosaic", "chr2\t200\tmosaic;cautious:AF<0.01"]


def test_stub_profile_inputs_exist():
    # F16: every ${projectDir}/... path of the test_stub profile is shipped.
    cfg = _text(os.path.join(ROOT, "tests", "nextflow", "test_stub.config"))
    paths = re.findall(r'"\$\{projectDir\}/([^"]+)"', cfg)
    assert len(paths) >= 15
    for p in paths:   # files, directories, or the VerifyBamID2 SVD prefix (<prefix>.UD, ...)
        assert os.path.exists(os.path.join(ROOT, p)) or os.path.exists(os.path.join(ROOT, p + ".UD")), p
    nfcfg = _text(os.path.join(ROOT, "nextflow.config"))
    assert "test_stub   { includeConfig 'tests/nextflow/test_stub.config' }" in nfcfg
    for sheet in ("samplesheet.csv", "samplesheet_fastq.csv"):
        base = os.path.join(ROOT, "tests", "nextflow")
        with open(os.path.join(base, sheet)) as fh:
            header = fh.readline().strip().split(",")
            for line in fh:
                row = dict(zip(header, line.rstrip("\n").split(",")))
                for col in ("bam", "fastq_1", "fastq_2"):
                    if row.get(col):
                        assert os.path.exists(os.path.join(base, row[col])), row[col]


@pytest.mark.skipif(not shutil.which("nextflow"), reason="nextflow not on PATH")
def test_nextflow_stub_run(tmp_path):
    t = os.path.join(ROOT, "tests", "nextflow")
    fastq = ["--samplesheet", os.path.join(t, "samplesheet_fastq.csv"), "--data_type", "wes",
             "--intervals", os.path.join(t, "resources", "targets.bed")]
    for i, extra in enumerate(([], fastq)):
        r = subprocess.run(
            ["nextflow", "-log", str(tmp_path / f"nf{i}.log"), "run", os.path.join(ROOT, "main.nf"), "-stub",
             "-profile", "test_stub", "--outdir", str(tmp_path / f"out{i}"), "-w", str(tmp_path / "work"),
             "-ansi-log", "false", *extra],
            cwd=tmp_path, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert (tmp_path / f"out{i}" / "pipeline_info" / "provenance.json").exists()

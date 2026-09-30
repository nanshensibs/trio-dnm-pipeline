import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=2000>
##contig=<ID=chrX,length=4000000>
##INFO=<ID=MQ,Number=1,Type=Float,Description="">
##INFO=<ID=FS,Number=1,Type=Float,Description="">
##INFO=<ID=SOR,Number=1,Type=Float,Description="">
##INFO=<ID=gnomAD_AF,Number=A,Type=Float,Description="">
##INFO=<ID=hiConfDeNovo,Number=1,Type=String,Description="">
##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description="">
"""


def write_fasta(path, seqs):
    with open(path, "w") as fh, open(path + ".fai", "w") as fai:
        offset = 0
        for name, seq in seqs.items():
            head = f">{name}\n"
            fh.write(head)
            offset += len(head)
            lines = [seq[i:i + 60] for i in range(0, len(seq), 60)]
            fai.write(f"{name}\t{len(seq)}\t{offset}\t60\t61\n")
            for ln in lines:
                fh.write(ln + "\n")
            offset += sum(len(ln) + 1 for ln in lines)


def gt(g, ref, alt, gq=99, pl=None, sb=None):
    dp = ref + alt
    if pl is None:
        # Approximate Q30 reads: each discordant read adds ~30 Phred, each read at a
        # het site adds ~3 Phred against a homozygous parent.
        pl = {"0/0": f"0,{3 * dp},{30 * dp}", "0/1": f"{30 * alt},0,{30 * ref}", "1/1": f"{30 * dp},{3 * dp},0"}.get(g, ".")
    sb = sb or f"{ref // 2},{ref - ref // 2},{alt // 2},{alt - alt // 2}"
    return f"{g}:{ref},{alt}:{dp}:{gq}:{pl}:{sb}"


@pytest.fixture
def trio_data(tmp_path):
    rng = random.Random(7)
    chr1 = [rng.choice("ACGT") for _ in range(2000)]
    # pos 100: C in a CpG (C at 100, G at 101)
    chr1[98], chr1[99], chr1[100] = "A", "C", "G"
    # homopolymer A x10 at 500-509
    for i in range(499, 509):
        chr1[i] = "A"
    chr1[509] = "C"
    for p in (200, 300, 400, 600, 700, 800, 900):
        if chr1[p - 1] not in "ACGT":
            chr1[p - 1] = "A"
    chrx = "".join(rng.choice("ACGT") for _ in range(4000))
    seqs = {"chr1": "".join(chr1), "chrX": "N" * 3_000_000 + chrx + "N" * (1_000_000 - 4000)}
    fasta = str(tmp_path / "ref.fa")
    write_fasta(fasta, seqs)

    def ref(chrom, pos):
        return seqs[chrom][pos - 1]

    alt_of = {"A": "G", "C": "T", "G": "A", "T": "C"}
    rows = []

    def add(chrom, pos, info, kid, dad, mom, alt=None):
        r = ref(chrom, pos)
        a = alt or alt_of[r]
        rows.append(f"{chrom}\t{pos}\t.\t{r}\t{a}\t500\tPASS\t{info}\tGT:AD:DP:GQ:PL:SB\t{kid}\t{dad}\t{mom}")

    good = "MQ=60;FS=1.2;SOR=0.7;hiConfDeNovo=kid"
    add("chr1", 100, good, gt("0/1", 15, 14), gt("0/0", 30, 0), gt("0/0", 32, 0))            # true DNM (CpG)
    add("chr1", 200, good, gt("0/1", 16, 14), gt("0/0", 28, 3, gq=40), gt("0/0", 30, 0))     # father leakage
    add("chr1", 300, good, gt("0/1", 15, 15), gt("0/0", 30, 0), gt("0/0", 40, 6, gq=30))     # maternal mosaic
    add("chr1", 400, "MQ=60;FS=1;SOR=0.7", gt("0/1", 45, 8), gt("0/0", 40, 0), gt("0/0", 42, 0))  # proband mosaic
    add("chr1", 600, good + ";gnomAD_AF=0.01", gt("0/1", 14, 15), gt("0/0", 30, 0), gt("0/0", 30, 0))  # common
    add("chr1", 700, good, gt("0/1", 14, 15), gt("0/0", 30, 0), gt("0/0", 30, 0))            # masked region
    add("chr1", 505, good, gt("0/1", 14, 15), gt("0/0", 30, 0), gt("0/0", 30, 0))            # homopolymer
    add("chr1", 800, good, gt("0/1", 14, 15), gt("0/1", 15, 15), gt("0/0", 30, 0))           # inherited
    add("chr1", 900, "MQ=30;FS=1;SOR=0.7", gt("0/1", 14, 15), gt("0/0", 30, 0), gt("0/0", 30, 0))  # low MQ
    add("chrX", 3_000_100, "MQ=60;FS=1;SOR=0.7", gt("1/1", 0, 16), gt("0/0", 18, 0), gt("0/0", 30, 0))  # male X hemi DNM
    vcf = tmp_path / "trio.vcf"
    vcf.write_text(HEADER + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom\n" + "\n".join(rows) + "\n")
    ped = tmp_path / "trio.ped"
    ped.write_text("F1\tkid\tdad\tmom\t1\t2\nF1\tdad\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n")
    bed = tmp_path / "mask.bed"
    bed.write_text("chr1\t690\t710\n")
    return {"vcf": str(vcf), "ped": str(ped), "fasta": fasta, "bed": str(bed), "tmp": tmp_path, "ref": ref}

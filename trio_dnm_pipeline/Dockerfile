# trio-dnm engine image for the Nextflow workflow (params.containers.trio_dnm).
#   docker build -t trio-dnm:2.0.0 .        (from trio_dnm_pipeline/)
# Python >= 3.10 runs the stdlib-only package and the bin/ helper scripts; procps
# provides `ps`, which Nextflow needs for task metrics (trace/report).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/trio-dnm
COPY pyproject.toml ./
COPY trio_dnm ./trio_dnm
# no dependencies; the image's own setuptools/wheel build it (no index access needed)
RUN pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /opt/trio-dnm/build
COPY bin/dng2tsv.py bin/vcf2mfbed.py /usr/local/bin/

WORKDIR /
CMD ["trio-dnm", "--help"]

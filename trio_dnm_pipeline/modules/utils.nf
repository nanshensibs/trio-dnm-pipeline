// Helper functions shared by the modules (included with `include { ... } from './utils'`).

// A staged `path` input as a list: one file arrives as a single path, several as a
// list and an omitted optional input ([]) as an empty list.
def asList(x) {
    x instanceof Collection ? x as List : (x ? [x] : [])
}

// The data file of a resource staged together with its index files (.tbi, .csi, ...).
def mainFile(x) {
    asList(x).find { !(it.name ==~ /.*\.(tbi|csi|idx|fai|gzi)$/) }
}

// Shell prelude for trio-dnm tasks: the package is staged into the task directory as
// pylib/trio_dnm (see nextflow.config), so it is importable in any python3 >= 3.10 image.
def trioDnmEnv() {
    'export PYTHONPATH="$PWD/pylib${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1'
}

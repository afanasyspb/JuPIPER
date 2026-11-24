# JuPIPER

A from-scratch Julia implementation of the mathematical core of
[PIPER](https://doi.org/10.1002/prot.21117), the FFT-based rigid-body
protein–protein docking algorithm (Kozakov, Brenke, Comeau & Vajda, *Proteins*
65:392–406, 2006). The numerical kernel is written once and runs on the CPU
(FFTW) or on an NVIDIA GPU (cuFFT) depending only on the array type it is given;
the scoring function is ordinary Julia code and can be changed without touching
a compiled language.

JuPIPER is a **clean-room** implementation: it is derived from the published
literature only. No source code of PIPER, ClusPro or any derivative has been
consulted.

## What it does, and what to expect

- Reads receptor and ligand structures from PDB files, assigns the 18 ACP atom
  types, builds the receptor's shape, electrostatic and pairwise-potential
  (DARS) fields on a grid, and correlates the ligand against them over a
  Hopf-fibration rotation set (70,000 rotations by default) with one FFT
  correlation per scoring term.
- Returns, for every rotation, the best translation and its energy; ranks the
  poses; clusters them by pairwise ligand RMSD; and reports native-pose rank,
  hits within a C-alpha RMSD threshold, and interface RMSD when a reference
  complex is available.
- Optional constraints from the ClusPro protocol: scoring profiles, residue
  masking, distance restraints and Cₙ-symmetric docking.

Two things to know before reading any result:

1. **The pipeline is validated; the scoring function is not calibrated.**
   Partial charges are zero by default (`--charges zero`), so the electrostatic
   term is inert, and the pairwise term uses the DARS potential alone. Docking a
   deposited complex against its own two chains recovers the native pose, which
   is what validates the machinery; it says nothing about unbound docking
   accuracy.
2. **The shape repulsion weight.** The 2006 paper publishes `w1 = 2.1` together
   with modified van der Waals radii it does not tabulate. With standard radii
   that weight makes the discretised shape term attractive inside the receptor
   core, and the search buries the ligand. The batch harness therefore defaults
   to `w1 = 7.0`; `scripts/validate_bound.jl` keeps the published `2.1` as its
   default so the effect can be reproduced. Pass `--w1` explicitly.

## Requirements

- **Julia 1.12.6** was used for release testing. The declared Julia compatibility range is 1.12 through 1.x.
- `bash`, `curl` and `sha256sum` for fetching structures (Git Bash on Windows).
- Optional: an NVIDIA GPU with a driver supported by CUDA.jl 6.

## Installation

```sh
git clone https://github.com/afanasyspb/JuPIPER.git
cd JuPIPER
julia --project=.    -e 'using Pkg; Pkg.instantiate()'
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=.    -t auto -e 'using Pkg; Pkg.test()'
```

The v0.1.0 CPU suite passed 5,210 checks with the following ten structures.
Fetch them to enable the real-structure tests.

```sh
bash scripts/fetch_structures.sh 1GGR 1F3G 1POH 1FGN 1TFH 1AVX 1BVN 1ACB 1AHW 4XAU
julia --startup-file=no --project=test -t 2 test/runtests.jl
```

Four project environments are provided. The root, test and bench environments include lockfiles.

| environment | contains | use it for |
|---|---|---|
| `--project=.` | the package | tests, library use without structure files |
| `--project=test` | package + BioStructures (the PDB reader) | **everything that opens a PDB file** |
| `--project=gpu` | package + BioStructures + CUDA | the same scripts on a GPU |
| `--project=bench` | package + benchmark utilities | numerical benchmarks without PDB input |

The PDB reader is a package extension that loads only when BioStructures is in
the environment, so a script that reads structures must run under `test` or
`gpu`.

## Quick start

### 1. Dock one complex against itself

```sh
bash scripts/fetch_structures.sh 1GGR
julia --project=test -t auto scripts/validate_bound.jl --case 1GGR --w1 7.0 --nrot 70000
```

Structures are downloaded from the RCSB PDB into `data/structures/` and checked
against the digests in `data/structures.sha256`; they are never committed. The
run prints the grid, the rotation set, the **search floor** (the best RMSD the
discrete search space contains, computed without any energy), and then the
rank of the native pose, the best pose found, and per-rank RMSDs. Read the
achieved RMSD against the floor, not against zero: the native orientation is
almost never a member of the rotation set.

Any bound complex works: `--complex path/to/file.pdb --rec-chain A --lig-chain B`.

### 2. Run the benchmark set

`data/benchmark51.tsv` is the enzyme category of Docking Benchmark 4.0
(52 cases over 51 PDB entries; 1OYV contributes two receptor chains). It is
generated from the benchmark's own supplementary table by
`scripts/fetch_benchmark4.jl`, which refuses to write the file if the table's
counts have changed.

```sh
bash scripts/fetch_structures.sh --bound      # the 52 complexes (~50 MB)
mkdir -p results
julia --project=test -t auto bench/benchmark51.jl --nrot 70000 --json results/benchmark51.json
```

The harness records one status per requested case. The JSON field `by_category`
separates enzyme/inhibitor recovery from the 1GGR control. `caselist_complete`
checks the identities and chains of all 52 benchmark cases. `run_complete` covers
every requested row, and `n_missing` and `n_error` distinguish missing inputs
from processing failures. Use `--limit 2` or
`--only 1AVX,1AY7` for a smoke test.

### 3. Run your own set of complexes

Write a tab-separated file with the same twelve columns as
`data/benchmark51.tsv` (lines starting with `#` are ignored):

```
case  complex  rec_chains  lig_chains  rec_unbound  rec_unbound_chains  lig_unbound  lig_unbound_chains  category  difficulty  resolution  note
```

Only `case`, `complex`, `rec_chains` and `lig_chains` are used for docking; put
`-` in the unbound columns if you have no unbound structures. Multi-chain sides
are written as concatenated one-letter chain ids (`AB`), and `.` is the blank
chain id of an old single-chain deposition. The harness opens
`<structures dir>/<complex>.pdb`, so local files that are not RCSB entries just
need that naming:

```sh
julia --project=test -t auto bench/benchmark51.jl \
      --cases my_cases.tsv --structures my_structures --json results/my_cases.json
```

`scripts/survey_caseset.jl --cases my_cases.tsv --structures my_structures`
reports, without docking anything, the grid each case needs and what it will
cost in memory and transforms.

### 4. Use the library from Julia

```julia
using BioStructures, JuPIPER, FFTW, LinearAlgebra
include("examples/docking.jl")           # write_complex
const T = Float64

pdb = "data/structures/1GGR.pdb"         # bash scripts/fetch_structures.sh 1GGR
rec = read_atoms(pdb; T = T, model = 1, chains = "A")
lig = read_atoms(pdb; T = T, model = 1, chains = "B")

tbl = load_acp_types()                  # the 18 ACP atom types
pot = load_dars()                       # the DARS pairwise potential
P   = dars_rank(pot)                    # eigen-rank used for the pairwise term
w   = PiperWeights{T}(7.0, PIPER_WEIGHTS_ENZYME_INHIBITOR.w2,
                           PIPER_WEIGHTS_ENZYME_INHIBITOR.w3)
par_rec = parametrize(tbl, rec)
par_lig = parametrize(tbl, lig)

spec = gridspec_for(rec, lig)           # grid size and 1.2 Å spacing from the two extents
FFTW.set_num_threads(1); BLAS.set_num_threads(1)   # parallelism is over rotations
rots = rotations_hopf_balanced(T, 72)

rs   = piper_receptor_spectra(JuPIPER.CPUBackend(), rec, par_rec, pot, spec; P = P, w = w)
ligS = piper_structure(lig, par_lig, pot; P = P)
pool = [workspace(JuPIPER.CPUBackend(), spec, 2 + P) for _ in 1:Threads.nthreads()]

poses = Vector{JuPIPER.Pose{T}}(undef, length(rots))
dock!(poses, ligS, rots.rots, rs, pool)  # best translation and energy per rotation

best = argmin(p -> p.energy, poses)
R, t = pose_transform(best, rots, spec)
write_complex("best.pdb", rec, lig, R, t)
```

Start Julia in the repository root with `julia --project=test -t auto`.
The writer adds `rec.centroid` to the centred receptor and the transformed
ligand. Chain R and chain L in `best.pdb` are both in the input PDB frame.
For the shorter listing in the paper, use `search_poses(rec, lig, rots; w3=3.0)`
from the same example file. The control uses `w1=7.0, w2=1500.0, w3=2.5`.

### 5. GPU

```sh
julia --project=gpu -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=gpu -t auto -e 'using CUDA, JuPIPER; include("test/runtests.jl")'   # adds the device tests
julia --project=gpu -t auto bench/benchmark51.jl --gpu --T float32 --json results/benchmark51-gpu.json
julia --project=gpu -t 1    bench/realdock.jl --T float32 --json results/realdock-1ggr.json
```

The harness selects a backend. The shared numerical loop dispatches on array type. `--gpu` makes the harness fail rather than silently run on the host, and
`--T float32` is the precision to use for docking on a device. `bench/realdock.jl`
docks one complex on both the host and the device in one process and reports
whether the two agree, cell for cell.

## Scripts

All options have defaults; the ones you are most likely to set are listed.

| script | purpose | options |
|---|---|---|
| `scripts/fetch_structures.sh` | fetch structures and verify published digests | `--verify` checks cached files. Select `--bound`, `--all` (153 IDs), individual IDs, or the default core set. `--fetch-new ID …` accepts unlisted IDs and prints their unverified hashes. |
| `scripts/fetch_benchmark4.jl` | regenerate `data/benchmark51.tsv` from the published table | `--out FILE` |
| `scripts/validate_bound.jl` | dock one complex bound–bound, report rank and RMSD | `--case`, `--complex`, `--rec-chain`, `--lig-chain`, `--nrot`, `--w1`, `--charges zero\|formal`, `--gb capped\|truncated`, `--P`, `--spacing`, `--pad`, `--hit`, `--topk`, `--ncluster`, `--cluster-radius`, `--batch`, `--json` |
| `bench/benchmark51.jl` | dock a case list | `--cases`, `--structures`, `--limit`, `--only`, `--nrot`, `--profile`, `--w1`, `--charges`, `--gb`, `--T float32\|float64`, `--hit`, `--topk`, `--cluster-radius`, `--max-clusters`, `--batch`, `--gpu`, `--json` |
| `scripts/survey_caseset.jl` | grid size and cost of a case list, no docking | `--cases`, `--structures`, `--json` |
| `bench/realdock.jl` | one complex on host and device, with a parity check | `--case`, `--rec-chain`, `--lig-chain`, `--nrot`, `--cpu-nrot`, `--T`, `--w1`, `--batch`, `--json` |
| `bench/multimer.jl` | Cₙ-symmetric docking of a homo-oligomer against the unconstrained search | `--case`, `--chain-a`, `--chain-b`, `--order`, `--tol`, `--nrot`, `--no-full`, `--json` |
| `scripts/validate_constraints.jl` | residue masking and distance restraints on the ClusPro protocol's two worked examples | `--arm mask\|restraint\|both`, `--nrot-mask`, `--nrot-restraint`, `--w1`, `--json` |
| `scripts/fetch_restraints.jl` | regenerate the E2A–HPr restraint file from cluspro.org | |

`--profile` selects one of the ClusPro 2017 scoring profiles from
`data/potentials/cluspro2017_profiles.tsv`; note that those coefficients belong
to a four-term energy and are carried over to this three-term one as an
assumption, which the file's header explains.

## Repository layout

```
src/            the package: grids, FFT correlation, rotation sets, scoring,
                PDB ingestion and atom typing, pose geometry, RMSD and
                clustering, masking, restraints, symmetry, device memory model
ext/            package extensions: the PDB reader (BioStructures) and the
                GPU backend (CUDA)
test/           the test suite; `test/Project.toml` is the environment that
                includes the PDB reader
bench/          the batch harness and the host/device docking scripts
scripts/        fetching data, one-complex validation, constraint validation
data/           DARS potential, ACP atom types, ClusPro profiles, the benchmark
                case list, structure digests, restraints and CDR definitions
gpu/            the CUDA environment (`gpu/Manifest.toml` is resolved locally)
examples/       the docking and PDB output example used in the paper
reproducibility/ scripts and LaTeX sources for the paper's tables and figures
results/wp8/    comparison measurements, summary and server-model digests
```

## Data provenance

The potential tables and benchmark list document their published sources.
`data/structures.sha256` pairs each RCSB entry with the digest of the PDB
bytes used here. The ACP atom assignments are reconstructed from the 2006
description. The ClusPro profiles apply published four-term coefficients to
a three-term energy. Both assumptions are described in the corresponding TSVs.

## Reproducing the paper

From the repository root, use Python 3.10 or later. The scripts read the
archived JSON measurements in `bench/results/` and `results/wp8/`. The
figure generator writes TikZ sources to `reproducibility/build/`. Install
pdfLaTeX with PGFPlots and the standalone class to render the PDFs, and
pdftocairo to also export SVGs.

```sh
python scripts/wp8_summary.py --print
python reproducibility/make_numbers.py
python reproducibility/make_figures.py
cd reproducibility
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build supplementary.tex
```

The CSV covers five ClusPro comparisons. Per-case server manifests list file
names, sizes and SHA-256 digests. Recalculating distances from coordinates
requires the original server models. The archived campaigns each contain 52
enzyme/inhibitor cases and the separate 1GGR control. Table S2 includes the
4CPA processing failure. The recovery curves include the completed no-hit
case 1D6R. The comments in `numbers.tex` identify 16 additional values
reported from measurements outside the archived JSON.

## References

- Kozakov D, Brenke R, Comeau SR, Vajda S. *PIPER: an FFT-based protein docking
  program with pairwise potentials.* Proteins 65(2):392–406, 2006.
  [doi:10.1002/prot.21117](https://doi.org/10.1002/prot.21117) — [arXiv:q-bio/0605018](https://arxiv.org/abs/q-bio/0605018).
- Padhorny D, Kazennov A, Zerbe BS, Porter KA, Xia B, Mottarella SE, Kholodov Y,
  Ritchie DW, Vajda S, Kozakov D. *Protein-protein docking by generalized Fourier
  transforms on 5D rotational manifolds.* [arXiv:1609.06743](https://arxiv.org/abs/1609.06743),
  PNAS 113(30):E4286–E4293, 2016. [doi:10.1073/pnas.1603929113](https://doi.org/10.1073/pnas.1603929113).
- Kozakov D et al. *The ClusPro web server for protein–protein docking.*
  Nature Protocols 12(2):255–278, 2017.
  [doi:10.1038/nprot.2016.169](https://doi.org/10.1038/nprot.2016.169).
- Hwang H, Vreven T, Janin J, Weng Z. *Protein–protein docking benchmark
  version 4.0.* Proteins 78(15):3111–3114, 2010.
- Desta IT, Porter KA, Xia B, Kozakov D, Vajda S. *Performance and its limits in
  rigid body protein–protein docking.* Structure 28(9):1071–1081, 2020.
  [doi:10.1016/j.str.2020.06.006](https://doi.org/10.1016/j.str.2020.06.006).

## Citation

Ilya Afanasyev (2026). *JuPIPER*, version 0.1.0. See [CITATION.cff](CITATION.cff).
<!-- Zenodo version DOI: insert the DOI for v0.1.0 here after archiving. -->

## License

JuPIPER is distributed under the [MIT License](LICENSE). Copyright © 2026 Ilya Afanasyev.
Third-party source publications and data retain their respective terms. See the references
above and the provenance notes in the data tables.

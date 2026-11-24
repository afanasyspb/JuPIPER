"""
    PDBSource

Backend tag for structure-file reading. Concrete subtypes name a parser; the
method that does the work lives wherever that parser is available.
"""
abstract type PDBSource end

"""
    BioStructuresPDB

Read through `BioStructures.jl`. The implementation is in the package extension
`JuPIPERBioStructuresExt` and is reached once `BioStructures` is loaded:

```julia
using BioStructures, JuPIPER
at = read_atoms("receptor.pdb")
```
"""
struct BioStructuresPDB <: PDBSource end

"""
Residue names treated as protein. Used only to disambiguate element inference
(see [`infer_element`](@ref)); atom *typing* is the business of `acp.jl` and is
driven by `data/potentials/acp18_types.tsv`, not by this set.

`MSE` is here because selenomethionine is a HETATM record but is protein.
"""
const PROTEIN_RESNAMES = Set([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",
])

"""
Modified residues kept despite arriving as HETATM records, mapped to the parent
residue they stand in for.

Deliberately short. `MSE` alone accounts for the overwhelming majority of
modified residues in crystal structures, and dropping it would silently delete a
buried hydrophobic side chain from every selenomethionine-phased structure --
which is a large fraction of everything solved since the late 1990s. Anything
not listed here is dropped by [`read_atoms`](@ref) **and counted by residue
name**, so a structure full of phosphoserine reports that fact rather than
quietly losing it.

Note the value is the parent for documentation and reporting only: the residue
keeps its own name, and `acp18_types.tsv` carries an explicit `MSE` block. That
avoids renaming `SE` to `SD` and pretending a selenium is a sulfur anywhere
except in the one place we say so (its ACP class).
"""
const MODIFIED_RESIDUES = Dict("MSE" => "MET", "SEC" => "CYS")

"Two-letter element symbols that occur in protein structures and common ligands."
const TWO_LETTER_ELEMENTS = Set([
    "SE", "FE", "ZN", "MG", "MN", "NA", "CL", "CU", "CO", "NI", "BR", "HG",
    "PT", "AU", "AG", "AL", "AS", "BA", "CS", "LI", "MO", "RB", "SR",

    "CA", "CD", "CE", "NE", "ND",
])

"""
Two-letter element symbols that are also the first two characters of a standard
protein atom name, and therefore cannot be inferred from the atom name alone.

Enumerated rather than guessed at. Protein heavy atoms are named from
`{N, CA, CB, CG, CD, CE, CZ, CH, OD, OE, OG, OH, ND, NE, NH, NZ, SD, SG, C, O,
OXT}`, and intersecting those prefixes with the periodic table gives exactly
`CA` (calcium / C-alpha), `CD` (cadmium / C-delta), `CE` (cerium / C-epsilon),
`NE` (neon / N-epsilon) and `ND` (neodymium / N-delta). Everything else --
`FE`, `ZN`, `MG`, `SE`, ... -- is unambiguous, which is why heme's `FE` can be
inferred inside a polyatomic `HEM` residue and `CA` cannot.
"""
const AMBIGUOUS_TWO_LETTER = Set(["CA", "CD", "CE", "NE", "ND"])

"""
    infer_element(resname, atomname) -> String

Recover an element symbol when the PDB `element` column is blank, which is
routine in files written before format v2.3 and not rare after it.

**This is not a cosmetic fallback -- it contains the nastiest naming collision
in the format.** `CA` is the alpha carbon in every amino acid and a *calcium
ion* in a `CA` HETATM residue. The two differ by 1.4x in van der Waals radius
and by an entire ACP class, and the atom name alone cannot tell them apart.

Three rules, in order:

1. **Inside a protein residue**, every atom is C, N, O or S named by its first
   alphabetic character. `SE` in selenomethionine is the one two-letter case.
2. **An unambiguous two-letter symbol** (`FE`, `ZN`, `MG`,...) is taken as such wherever
   it appears, so heme's iron resolves correctly.
3. **An ambiguous two-letter symbol** ([`AMBIGUOUS_TWO_LETTER`](@ref)) is taken
   as an element **only when the residue is monatomic**, i.e. `resname ==
   atomname`, which is how the PDB writes a bare ion. Otherwise it falls back
   to the first character.

Getting it wrong in the permissive direction turns a modified residue's alpha carbon --
`SEP`, `TPO`, `PTR` all have one -- into calcium, and assigns it a *plausible* radius
that no downstream check can question. Getting it wrong in the strict direction gives a
`CD` cadmium ion a carbon radius, which `parametrize` cannot see either, but which
requires a bare cadmium in a polyatomic residue with a blank element column.

Leading digits (`1HB`, `2HG1`) are skipped. Returns `""` if nothing can be
recovered. Allocates a small string; setup-time.
"""
function infer_element(resname::AbstractString, atomname::AbstractString)
    an = uppercase(strip(atomname))
    isempty(an) && return ""
    i = firstindex(an)
    while i <= lastindex(an) && isdigit(an[i])
        i = nextind(an, i)
    end
    i > lastindex(an) && return ""
    core = String(SubString(an, i))
    if uppercase(strip(resname)) in PROTEIN_RESNAMES
        core == "SE" && return "SE"
        return string(core[1])
    end
    if length(core) >= 2
        two = core[1:2]
        if two in TWO_LETTER_ELEMENTS
            two in AMBIGUOUS_TWO_LETTER || return two
            uppercase(strip(resname)) == an && return two
        end
    end
    return string(core[1])
end

"""
    IngestReport

Everything [`read_atoms`](@ref) discarded or repaired, and why.

This exists because the failure mode of structure parsing is not an exception,
it is a quiet wrong answer: a chain silently missing, a selenomethionine
deleted, an alternate conformation chosen without saying so. Each field is a
count, so the caller can assert on them. `show` prints the non-zero ones.
"""
struct IngestReport
    path::String
    models_present::Int
    model_selected::Int
    chains_present::Vector{String}
    chains_kept::Vector{String}
    natoms_seen::Int
    natoms_kept::Int
    n_hydrogen::Int
    n_water::Int
    n_hetero_dropped::Int
    hetero_resnames::Dict{String,Int}
    n_modified_kept::Int
    modified_resnames::Dict{String,Int}
    n_altloc::Int
    n_element_inferred::Int
    n_residues::Int
    n_cterm_complete::Int
    n_cterm_no_oxt::Int
end

function Base.show(io::IO, r::IngestReport)
    print(io, "IngestReport(", basename(r.path), ": ", r.natoms_kept, "/",
          r.natoms_seen, " atoms kept, ", r.n_residues, " residues, chains ",
          join(r.chains_kept, ","))
    r.models_present > 1 && print(io, ", model ", r.model_selected, "/", r.models_present)
    r.n_hydrogen > 0        && print(io, ", -", r.n_hydrogen, "H")
    r.n_water > 0           && print(io, ", -", r.n_water, " water")
    r.n_hetero_dropped > 0  && print(io, ", -", r.n_hetero_dropped, " hetero")
    r.n_modified_kept > 0   && print(io, ", +", r.n_modified_kept, " modified")
    r.n_altloc > 0          && print(io, ", ", r.n_altloc, " altloc")
    r.n_element_inferred > 0 && print(io, ", ", r.n_element_inferred, " elem inferred")
    r.n_cterm_no_oxt > 0    && print(io, ", ", r.n_cterm_no_oxt, " C-term w/o OXT")
    print(io, ")")
end

"""
    AtomTable{T}

Heavy atoms of one docking partner, struct-of-arrays.

`coords` is the only field the numerical pipeline reads; the rest is identity
and provenance, kept so that parametrisation (`acp.jl`) and diagnostics have
something to work from and so a pose can be written back out.

Adding `centroid` restores the original crystal frame after [`center!`](@ref).
This permits comparison of a docked pose with the deposited complex.

`is_nterm` / `is_cterm` are per-atom flags for "belongs to a chain-terminal
residue". `is_cterm` additionally requires the residue to carry an `OXT`, i.e.
it means *chemically complete C-terminus*, not merely *last residue present*.
A chain truncated by missing density is not a carboxylate and must not be
charged like one; the count of such cases is in the report.
"""
mutable struct AtomTable{T<:AbstractFloat}
    coords::Vector{SVector{3,T}}
    element::Vector{String}
    atomname::Vector{String}
    resname::Vector{String}
    resnum::Vector{Int32}
    inscode::Vector{Char}
    chain::Vector{String}
    occupancy::Vector{T}
    tempfactor::Vector{T}
    is_nterm::BitVector
    is_cterm::BitVector
    centroid::SVector{3,T}
    centred::Bool
    report::IngestReport
end

Base.length(at::AtomTable) = length(at.coords)
natoms(at::AtomTable) = length(at.coords)
Base.eltype(::AtomTable{T}) where {T} = T

function Base.show(io::IO, at::AtomTable{T}) where {T}
    print(io, "AtomTable{", T, "}(", length(at), " atoms, ",
          at.centred ? "centred" : "crystal frame", ")")
end

"""
    center!(at) -> at

Subtract the centroid of `at.coords` in place and record it in `at.centroid`.

Mutates `at`. Allocates nothing. Idempotent: calling it on an already-centred
table is a no-op, because the centroid of centred coordinates is zero and the
accumulated shift is what `at.centroid` stores.

Centring is not cosmetic. `Structure` documents that its coordinates are
centroid-centred so that a rotation is a pure `R * x` with no translation
bookkeeping, and so that the closed-form RMSD in `pairwise_rmsd!` is valid.
"""
function center!(at::AtomTable{T}) where {T}
    n = length(at.coords)
    n == 0 && return at
    acc = zero(SVector{3,T})
    @inbounds for i in 1:n
        acc += at.coords[i]
    end
    c = acc / T(n)
    @inbounds for i in 1:n
        at.coords[i] -= c
    end
    at.centroid += c
    at.centred = true
    return at
end

"""
    uncenter!(at) -> at

Undo [`center!`](@ref): put the coordinates back in the frame they were read in.
Mutates `at`; allocates nothing.
"""
function uncenter!(at::AtomTable{T}) where {T}
    c = at.centroid
    @inbounds for i in eachindex(at.coords)
        at.coords[i] += c
    end
    at.centroid = zero(SVector{3,T})
    at.centred = false
    return at
end

"""
    extent(at) -> SVector{3,T}

Bounding-box side lengths, in angstrom. Allocates nothing.
"""
function extent(at::AtomTable{T}) where {T}
    isempty(at.coords) && return zero(SVector{3,T})
    lo = at.coords[1]
    hi = at.coords[1]
    @inbounds for i in 2:length(at.coords)
        lo = min.(lo, at.coords[i])
        hi = max.(hi, at.coords[i])
    end
    return hi - lo
end


"""
    TailReport{T}

What a structure's **extent** costs, as against what its atoms weigh.

This exists because of one line in the 2017 ClusPro protocol's troubleshooting
table (published p.271, entry 10): *"Persistant crashing of job when the input
file looks fine. FFT grids are too large. This is typically caused by uploading
a file output by homology modeling that has extremely large regions without a
template, resulting in a tail that basically floats off into space."*

The failure is a **grid-size** failure caused by a handful of atoms carrying no
information, and its published symptom is a crash with an unhelpful cause. So this is
the same question asked one step earlier, with no grid in hand: **how much of the
enclosing sphere is paid for by how few atoms?**

The statistic is the **enclosing radius against the radius of gyration**, and
choosing it that way is the point. A percentile-based "core radius" cannot work
here: a homology-model tail is often 10-15% of the atoms, so any percentile
loose enough to ignore one flexible terminus is also loose enough to sit *inside*
the tail and report nothing. `Rg` has no such cutoff -- it is an average, so a
tail moves `r_max` far more than it moves `Rg`, whatever fraction of the
structure the tail is.

`ratio = r_max / Rg` is `sqrt(5/3) = 1.291` for a uniform ball, and
`inflation = (ratio / sqrt(5/3))^3` is therefore the factor by which the sphere
that must hold every rotation of this structure exceeds the sphere a compact body
of the same spread would need. It is a cell count, since spacing is fixed: an
inflation of 20 is twenty times the transform work for the same protein. The
sphere rather than the bounding box is deliberate and is not an approximation of
it -- `grid_fit_rotated` bounds every rotation by the enclosing sphere precisely
so that it need not enumerate them, so the sphere *is* what a rotated ligand's
grid is sized by.

**The threshold is calibrated, not chosen.** Measured on this repository's own
structures, `ratio` is 1.68-1.85 for every single deposited assembly tried
(1GGR, 1F3G, 1POH, 1FGN, 1TFH, 1AVX, 1BVN, 1ACB, 1KKL), i.e. `inflation` 2.2-2.95;
the one outlier is 1AHW at 2.14 / 4.58, and that is a *true* positive -- the file
holds two complete copies of the complex, so reading all six chains as one body
really would size a grid for a barbell. Synthetic globule-plus-tail cases span
`inflation` 16-67 across tails of 1%, 5%, 13% and 23% of the atoms. So
[`TAIL_INFLATION_WARN`](@ref) sits at 8: comfortably above the multi-copy case,
comfortably below the loosest tail.

Detection only.
"""
struct TailReport{T<:AbstractFloat}
    natoms::Int
    rg::T
    r_max::T
    ratio::T
    inflation::T
    worst_atom::Int
    worst_label::String
end

function Base.show(io::IO, r::TailReport{T}) where {T}
    print(io, "TailReport{$T}: ", r.natoms, " atoms, Rg ",
          round(r.rg; digits = 2), " A, r_max ", round(r.r_max; digits = 2),
          " A, r_max/Rg ", round(r.ratio; digits = 3), ", grid inflation ",
          round(r.inflation; digits = 2), "x")
    r.natoms > 0 && print(io, ", farthest ", r.worst_label)
end

"""Uniform ball: `r_max / Rg` = `sqrt(5/3)`. The reference shape, not a fudge."""
const BALL_RATIO = sqrt(5 / 3)

"""
    tail_report(at) -> TailReport

Characterise how far a structure's extent runs past its bulk. Allocates nothing
beyond the report; setup-time, called once per partner by [`read_atoms`](@ref).

Radii are measured from the structure's own centroid whether or not the table
has been centred, so the answer does not depend on the frame.
"""
function tail_report(at::AtomTable{T}) where {T}
    n = length(at.coords)
    n == 0 && return TailReport{T}(0, zero(T), zero(T), zero(T), one(T), 0, "")
    c = sum(at.coords) / T(n)
    s2 = zero(T); r_max = zero(T); worst = 1
    @inbounds for i in eachindex(at.coords)
        d2 = sum(abs2, at.coords[i] - c)
        s2 += d2
        if d2 > r_max
            r_max = d2; worst = i
        end
    end
    rg = sqrt(s2 / T(n))
    r_max = sqrt(r_max)

    ratio = rg > 0 ? r_max / rg : T(BALL_RATIO)
    infl = (ratio / T(BALL_RATIO))^3
    label = string(at.chain[worst], " ", at.resname[worst], at.resnum[worst],
                   " ", at.atomname[worst])
    return TailReport{T}(n, rg, r_max, ratio, infl, worst, label)
end

"""
    warn_on_tail(at, report; threshold = 2.0) -> Bool

**The threshold is ours, not published** -- the protocol names the failure and
its cause but gives no number -- so it is derived from measurement rather than
picked: see [`TailReport`](@ref) for the calibration that puts every real
structure here below 4.6 and every synthetic tail above 16. It is a named
constant rather than an inline literal for the reason the definition-of-done
gives, and it is deliberately far above the largest observed true structure so
that a default-on warning cannot become noise a user learns to ignore.
"""
const TAIL_INFLATION_WARN = 8.0

function warn_on_tail(at::AtomTable, r::TailReport;
                      threshold::Real = TAIL_INFLATION_WARN)
    r.inflation >= threshold || return false
    @warn """
        This structure's extent runs far past its bulk: r_max/Rg is \
        $(round(r.ratio; digits = 2)) against $(round(BALL_RATIO; digits = 2)) for a compact body, so the \
        enclosing sphere -- and hence the FFT grid -- is about \
        $(round(r.inflation; digits = 0))x larger than this much protein needs.
        $(r.natoms) atoms, Rg $(round(r.rg; digits = 1)) A, farthest atom $(r.worst_label) at \
        $(round(r.r_max; digits = 1)) A.
        This is the signature of an unstructured tail -- typically a homology model \
        with a long region that had no template, or a file holding more copies of \
        the assembly than you meant to read. The 2017 ClusPro protocol reports it \
        as "persistant crashing ... FFT grids are too large" (published p.271). \
        Select fewer chains, trim the tail, or pass `check_tail = false` if the \
        extent is real."""
    return true
end


function _read_atoms(src::PDBSource, path::AbstractString; kwargs...)
    throw(ArgumentError("""
        no PDB reader is loaded for $(typeof(src)).

        `read_atoms` is supplied by a package extension, so the parser is not a
        dependency of JuPIPER itself. Load it first:

            using BioStructures, JuPIPER

        (in a project that has BioStructures; `test/Project.toml` and
        `gpu/Project.toml` already do.)"""))
end

"""
    read_atoms(path; kwargs...) -> AtomTable

Read one docking partner from a structure file.

Requires a reader to be loaded -- see [`BioStructuresPDB`](@ref). Allocates;
setup-time, called twice per docking job.

# Keywords
- `T = Float64`: coordinate element type.
- `source = BioStructuresPDB()`: which parser to dispatch to.
- `chains = nothing`: keep only these chain ids (a string or a collection of
  them). `nothing` keeps all.
- `model = nothing`: which model to read. `nothing` takes the lowest-numbered
  one, which is the convention for NMR ensembles.
- `keep_hydrogens = false`: ACP and DARS are heavy-atom schemes and their
  contact statistics were counted over heavy atoms, so hydrogens are noise here
  rather than detail.
- `keep_water = false`.
- `keep_hetero = false`: general HETATM records -- ligands, ions, cryoprotectant.
  Dropped by default *and counted by residue name*. Modified residues in
  [`MODIFIED_RESIDUES`](@ref) are kept regardless.
- `center = true`: subtract the centroid, recording it (see [`center!`](@ref)).

# What it will not do silently
Every atom that is discarded is counted in `at.report`, by reason and -- for
HETATM -- by residue name. A structure whose ligand you meant to keep does not
come back short with no explanation.
"""
function read_atoms(path::AbstractString; source::PDBSource = BioStructuresPDB(),
                    check_tail::Bool = true, kwargs...)
    at = _read_atoms(source, path; kwargs...)

    check_tail && warn_on_tail(at, tail_report(at))
    return at
end

"""
    ACP18_LABELS

The 18 ACP type labels, **in the order the DARS matrix rows and columns use**.

That order is load-bearing rather than cosmetic: a type index produced here is
used directly to index `Potential.U`, so a permutation would silently pair every
atom with the wrong eigenvector component and produce a plausible, wrong energy.
`test_pdb.jl` pins this against `load_dars().labels`.

Greek letters are spelled out: z = zeta, d = delta, h = eta, e = epsilon,
g = gamma, b = beta.
"""
const ACP18_LABELS = ["N", "Ca", "C", "O", "GCa", "Cb", "KNz", "KCd", "DOd",
                      "RNh", "NNd", "RNe", "SOg", "HNe", "YCz", "FCz", "LCd", "CSg"]

"""
Number of distinct heavy atoms in the 20 standard amino acids.

Not a magic number: the PIPER preprint states that "the existing pairwise
interaction potentials can have up to 167 atom types" (PDF p.12), which is this
count. It is the completeness invariant on `acp18_types.tsv` -- if a residue
loses or gains an atom in transcription, the total moves off 167 and
`test_pdb.jl` fails.
"""
const ACP_STANDARD_ATOM_COUNT = 167

"""
    ACPTypeTable

The (residue, atom) -> ACP type index lookup, plus the label vocabulary.

`anyres` holds rows whose residue field is `*` and is consulted only when no
exact match exists. It exists for `OXT`, which every residue can carry and none
carries in its own right.
"""
struct ACPTypeTable
    labels::Vector{String}
    index::Dict{String,Int8}
    byres::Dict{String,Dict{String,Int8}}
    anyres::Dict{String,Int8}
    path::String
end

Base.show(io::IO, t::ACPTypeTable) =
    print(io, "ACPTypeTable(", length(t.labels), " types, ",
          sum(length, values(t.byres); init = 0), " (residue,atom) rows, ",
          length(t.anyres), " wildcard)")

"""
    nresidues(t::ACPTypeTable)

Number of residue names the table knows. Allocates nothing.
"""
nresidues(t::ACPTypeTable) = length(t.byres)

"""
    load_acp_types(path=...; labels=ACP18_LABELS) -> ACPTypeTable

Read the ACP 18-type classification from a TSV.

See `data/potentials/acp18_types.tsv` for provenance -- **the table is a
reconstruction, not a transcription**, and the header explains why and lists the
places it is most likely to be wrong. Allocates; setup-time.

Throws if the file names a type label that is not in `labels`, which is the
check that keeps the file and [`ACP18_LABELS`](@ref) from drifting apart.
"""
function load_acp_types(path::AbstractString = joinpath(@__DIR__, "..", "data",
                                                        "potentials",
                                                        "acp18_types.tsv");
                        labels::AbstractVector{<:AbstractString} = ACP18_LABELS)
    index = Dict{String,Int8}(String(l) => Int8(i) for (i, l) in enumerate(labels))
    byres = Dict{String,Dict{String,Int8}}()
    anyres = Dict{String,Int8}()
    lineno = 0
    for line in eachline(path)
        lineno += 1
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s, '\t')
        length(f) == 3 || throw(ArgumentError(
            "$path:$lineno: expected 3 tab-separated fields, got $(length(f))"))
        resn, an, lab = uppercase(strip(f[1])), uppercase(strip(f[2])), String(strip(f[3]))
        haskey(index, lab) || throw(ArgumentError(
            "$path:$lineno: unknown ACP type label $(repr(lab)); known: $(labels)"))
        ti = index[lab]
        if resn == "*"
            anyres[an] = ti
        else
            sub = get!(() -> Dict{String,Int8}(), byres, resn)
            haskey(sub, an) && throw(ArgumentError(
                "$path:$lineno: duplicate row for ($resn, $an)"))
            sub[an] = ti
        end
    end
    return ACPTypeTable(String.(labels), index, byres, anyres, String(path))
end

"""
    acp_type(tbl, resname, atomname) -> Int

ACP type index in `1:18`, or `0` if the pair is not in the table.

Zero is a deliberate, checkable sentinel rather than a fallback type: an atom we
cannot classify must not be silently assigned to whichever class happens to be
nearest, because the resulting energy would be wrong in a way no test can see.
[`parametrize`](@ref) collects the zeros and reports them. Allocates nothing.
"""
@inline function acp_type(tbl::ACPTypeTable, resname::AbstractString,
                          atomname::AbstractString)
    sub = get(tbl.byres, resname, nothing)
    if sub !== nothing
        t = get(sub, atomname, Int8(0))
        t != 0 && return Int(t)
    end
    return Int(get(tbl.anyres, atomname, Int8(0)))
end

"""
    acp_types!(out, tbl, resnames, atomnames) -> out

Bulk form of [`acp_type`](@ref). Mutates `out`; **allocates nothing** after warm-up,
which `test_pdb.jl` asserts from inside a function with concrete argument types.
"""
function acp_types!(out::AbstractVector{<:Integer}, tbl::ACPTypeTable,
                    resnames::AbstractVector{<:AbstractString},
                    atomnames::AbstractVector{<:AbstractString})
    n = length(out)
    (length(resnames) == n && length(atomnames) == n) ||
        throw(DimensionMismatch("out, resnames and atomnames must agree in length"))
    @inbounds for i in 1:n
        out[i] = acp_type(tbl, resnames[i], atomnames[i])
    end
    return out
end


"""
    has_vdw_radius(element) -> Bool

Whether [`VDW_RADII`](@ref) carries this element, i.e. whether
[`vdw_radius`](@ref) will return a real value rather than its carbon default.

Exists so that the default can be *reported* instead of applied silently: a
structure containing an element we have no radius for should say so, not quietly
model it as carbon.
"""
has_vdw_radius(element::AbstractString) = haskey(VDW_RADII, uppercase(strip(element)))


"""
    ChargeModel

How the `q_i` of PIPER's electrostatic term are assigned.

The preprint's `E_elec` is a simplified Generalized Born expression with
constant Born radii (PDF p.10-11):

    R_p(l,m,n) = sum_i  q_i / (r^2 + D^2 exp(-r^2 / 4D^2))^(1/2),  r = max(r, 2D)

and it never says where `q_i` comes from. In the reference lineage those are
CHARMM charges -- roughly 170 numbers spread over the 20 residues.

So the charge assignment is a seam with two implementations, and the honest one is the
default:

  * [`ZeroCharges`](@ref) -- **the default.** An explicit abstention. The
    electrostatic term evaluates to zero and is visibly inert, rather than
    quietly wrong.
  * [`FormalCharges`](@ref) -- a placeholder that is *derivable rather than
    recalled*.
"""
abstract type ChargeModel end

"""
    ZeroCharges

Every atom carries `q = 0`. The default, and an abstention rather than a guess:
see [`ChargeModel`](@ref).
"""
struct ZeroCharges <: ChargeModel end

"""
    FormalCharges

Textbook formal charge at pH 7, placed on the ionisable groups and nowhere else.

Not a force field and not a substitute for one. It has no dipoles at all -- a
backbone carbonyl, the single largest contributor to real protein
electrostatics, is neutral here -- so it can represent salt bridges and net
monopole attraction and nothing finer.

It is included because it is **derivable from chemistry rather than recalled
from a table**, so it can be audited on inspection:

| group                | charge                        |
|----------------------|-------------------------------|
| ASP OD1, OD2         | -1/2 each (delocalised)       |
| GLU OE1, OE2         | -1/2 each                     |
| LYS NZ               | +1                            |
| ARG NE, NH1, NH2     | +1/3 each (guanidinium)       |
| N-terminal N         | +1                            |
| C-terminal O, OXT    | -1/2 each                     |
| HIS                  | 0 -- neutral at pH 7          |

The C-terminal pair is charged only when the terminus is chemically complete,
i.e. when `OXT` is present; a chain truncated by missing density is not a
carboxylate. `AtomTable.is_cterm` already encodes that distinction.
"""
struct FormalCharges <: ChargeModel end

chargemodel_name(::ZeroCharges) = :zero
chargemodel_name(::FormalCharges) = :formal

atom_charge(::ZeroCharges, ::Type{T}, ::AbstractString, ::AbstractString,
            ::Bool, ::Bool) where {T} = zero(T)

function atom_charge(::FormalCharges, ::Type{T}, resname::AbstractString,
                     atomname::AbstractString, nterm::Bool, cterm::Bool) where {T}
    q = zero(T)
    if resname == "ASP" && (atomname == "OD1" || atomname == "OD2")
        q -= T(0.5)
    elseif resname == "GLU" && (atomname == "OE1" || atomname == "OE2")
        q -= T(0.5)
    elseif resname == "LYS" && atomname == "NZ"
        q += one(T)
    elseif resname == "ARG" &&
           (atomname == "NE" || atomname == "NH1" || atomname == "NH2")
        q += one(T) / T(3)
    end

    nterm && atomname == "N" && (q += one(T))
    cterm && (atomname == "O" || atomname == "OXT") && (q -= T(0.5))
    return q
end


"""
    PiperWeights{T}

The three weights of `E = E_shape + w2 * E_elec + w3 * E_pair`, with
`E_shape = E_attr + w1 * E_rep` (PIPER preprint, PDF p.9).

**These are published values, not placeholders.** The preprint fits them by
logistic regression on a small subset of the benchmark and reports them
outright (PDF p.15):

  * enzyme-inhibitor: `w1 = 2.1, w2 = 133.0, w3 = 2.2`
  * antibody-antigen: `w1 = 2.0, w2 = 400.0, w3 = 1.0`

Worth knowing why that matters here: `shape.jl` carried `w_rep = 9` as a "placeholder of
the right order" under a docstring asserting that the paper "deliberately does not
publish a value". The assertion was reading p.9, which describes `w1` qualitatively, and
missing p.15, which gives it.
"""
struct PiperWeights{T<:AbstractFloat}
    w1::T
    w2::T
    w3::T
end

PiperWeights{T}(w::PiperWeights) where {T} = PiperWeights{T}(T(w.w1), T(w.w2), T(w.w3))

"Weights fitted by the preprint for enzyme-inhibitor complexes (PDF p.15)."
const PIPER_WEIGHTS_ENZYME_INHIBITOR = PiperWeights{Float64}(2.1, 133.0, 2.2)

"Weights fitted by the preprint for antibody-antigen complexes (PDF p.15)."
const PIPER_WEIGHTS_ANTIBODY_ANTIGEN = PiperWeights{Float64}(2.0, 400.0, 1.0)

"""
    DARS_ACP_MIX

The preprint's production pairwise term is a linear combination of two matrices
(PDF p.14):

    E_{DARS+ACP} = (3 * E_DARS + 0.5 * E_ACP) / 4.0

JuPIPER uses the DARS matrix alone. The reference combination above also
requires the ACP matrix of Zhang et al.
"""
const DARS_ACP_MIX = (dars = 3.0 / 4.0, acp = 0.5 / 4.0)


"""
    Parametrization{T}

Per-atom physics for one [`AtomTable`](@ref): ACP type index, partial charge,
van der Waals radius, and the record of everything that could not be assigned.

`untyped` and `unknown_element` hold **indices into the atom table**, not
counts, so a caller can look at the offending atoms rather than guess.
"""
struct Parametrization{T<:AbstractFloat}
    acptype::Vector{Int8}
    charge::Vector{T}
    radius::Vector{T}
    labels::Vector{String}
    untyped::Vector{Int32}
    unknown_element::Vector{Int32}
    chargemodel::Symbol
    net_charge::T
    terminal_carboxylate::Bool
end

Base.length(p::Parametrization) = length(p.acptype)

function Base.show(io::IO, p::Parametrization{T}) where {T}
    print(io, "Parametrization{", T, "}(", length(p), " atoms, charges=",
          p.chargemodel, ", net q=", round(p.net_charge; digits = 2))
    isempty(p.untyped) || print(io, ", ", length(p.untyped), " UNTYPED")
    isempty(p.unknown_element) ||
        print(io, ", ", length(p.unknown_element), " default radius")
    print(io, ")")
end

"""
    type_histogram(p) -> Dict{String,Int}

Atom count per ACP type label. Allocates; diagnostic.
"""
function type_histogram(p::Parametrization)
    h = Dict{String,Int}()
    for t in p.acptype
        t == 0 && continue
        l = p.labels[t]
        h[l] = get(h, l, 0) + 1
    end
    return h
end

"""
    parametrize(tbl, at; kwargs...) -> Parametrization

Assign ACP types, van der Waals radii and partial charges to an
[`AtomTable`](@ref). Allocates; setup-time.

# Keywords
- `charges = ZeroCharges`: see [`ChargeModel`](@ref). The default assigns zero to every
  atom, deliberately.
- `terminal_carboxylate = false`: promote the C-terminal backbone `O` from type `O` to
  `DOd`, so that both halves of a complete C-terminal carboxylate carry the carboxylate
  type. One atom per chain either way.
- `on_untyped = :warn`: `:warn`, `:error` or `:ignore`. Untyped atoms keep type
  `0` and are listed in the result regardless -- this only controls whether you
  hear about it.

# What it will not do
It will not guess. An atom whose (residue, atom) pair is absent from the table
gets type `0` and appears in `untyped`; an element absent from
[`VDW_RADII`](@ref) gets the carbon default *and* appears in `unknown_element`.
Both are recoverable by the caller; neither is silent.
"""
function parametrize(tbl::ACPTypeTable, at::AtomTable{T};
                     charges::ChargeModel = ZeroCharges(),
                     terminal_carboxylate::Bool = false,
                     on_untyped::Symbol = :warn) where {T}
    on_untyped in (:warn, :error, :ignore) ||
        throw(ArgumentError("on_untyped must be :warn, :error or :ignore"))
    n = length(at)
    types = Vector{Int8}(undef, n)
    acp_types!(types, tbl, at.resname, at.atomname)

    if terminal_carboxylate
        dod = tbl.index["DOd"]
        @inbounds for i in 1:n
            if at.is_cterm[i] && at.atomname[i] == "O"
                types[i] = dod
            end
        end
    end

    q = Vector{T}(undef, n)
    r = Vector{T}(undef, n)
    untyped = Int32[]
    unknown_el = Int32[]
    @inbounds for i in 1:n
        types[i] == 0 && push!(untyped, Int32(i))
        has_vdw_radius(at.element[i]) || push!(unknown_el, Int32(i))
        r[i] = vdw_radius(T, at.element[i])
        q[i] = atom_charge(charges, T, at.resname[i], at.atomname[i],
                           at.is_nterm[i], at.is_cterm[i])
    end

    if !isempty(untyped) && on_untyped !== :ignore

        pairs = Dict{String,Int}()
        for i in untyped
            k = string(at.resname[i], " ", at.atomname[i])
            pairs[k] = get(pairs, k, 0) + 1
        end
        msg = "$(length(untyped)) atom(s) have no ACP type in $(basename(tbl.path))"
        if on_untyped === :error
            throw(ArgumentError("$msg: $(sort!(collect(pairs); by = last, rev = true))"))
        else
            @warn msg pairs = sort!(collect(pairs); by = last, rev = true)
        end
    end

    return Parametrization{T}(types, q, r, tbl.labels, untyped, unknown_el,
                              chargemodel_name(charges), sum(q; init = zero(T)),
                              terminal_carboxylate)
end

"""
    piper_structure(at, par, pot; P=dars_rank(pot)) -> Structure

Builds the `(natoms, 2 + P)` term-weight matrix in PIPER's term order via
[`build_termweights`](@ref) -- shape, electrostatics, then the `P` dominant
DARS eigenvector components -- and pairs it with the centred coordinates.

Allocates; setup-time. Throws if any atom is untyped (a type index of `0` would
index `pot.U` out of bounds, so this is a bounds error caught early and named)
or if the table has not been centred, which `Structure` requires.

`P` is **derived from the matrix, not fixed**: it defaults to
[`dars_rank`](@ref)`(pot)`, the smallest rank whose truncation error meets the
10% the preprint uses to justify truncating at all (PDF p.13). On DARS that is
5. The tempting literal is 4 -- the top of the preprint's "2 to 4 eigenvalues suffice"
   -- but that sentence is about ACP and RPScore, not about DARS, and rank 4 measures
   14.5% here.
"""


function piper_structure(at::AtomTable{T}, par::Parametrization{T},
                         pot::Potential{T}; P::Integer = dars_rank(pot)) where {T}
    length(par) == length(at) ||
        throw(DimensionMismatch("parametrization and atom table disagree in length"))
    at.centred || throw(ArgumentError(
        "coordinates must be centred before building a Structure; call center!(at)"))
    isempty(par.untyped) || throw(ArgumentError(
        "$(length(par.untyped)) atom(s) are untyped; a type index of 0 cannot " *
        "index the eigenvector matrix. Fix the table or drop the atoms."))
    K = size(pot.U, 1)
    length(pot.labels) == length(par.labels) && pot.labels == par.labels ||
        throw(ArgumentError("potential labels and type-table labels disagree; " *
                            "the type index would select the wrong eigenvector"))
    1 <= P <= K || throw(ArgumentError("P must be in 1:$K, got $P"))
    W = build_termweights(par.acptype, par.charge, view(pot.U, :, 1:P))
    return Structure(copy(at.coords), W)
end

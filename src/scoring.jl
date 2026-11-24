"""
    ClusProProfile{T}

One row of the 2017 ClusPro protocol's Table 1: the four weighting coefficients
of `E = w_rep*E_rep + w_attr*E_attr + w_elec*E_elec + w_dars*E_DARS`, in the
paper's own parameterisation, plus the paper's own name for the mode.

**These are published values for an energy function JuPIPER does not
implement.** Held in their published form deliberately: converting on the way
in would bury the assumption that the conversion rests on. See
[`assumed_piper_weights`](@ref) for the conversion and what it assumes, and
`data/potentials/cluspro2017_profiles.tsv` for the provenance.

`w_attr` is negative in every published set. That is not a sign convention we
imposed: `E_attr` is a positive quantity there, as our contact count is here,
and the negative weight is what makes contact favourable.
"""
struct ClusProProfile{T<:AbstractFloat}
    key::Symbol
    label::String
    w_rep::T
    w_attr::T
    w_elec::T
    w_dars::T
end

ClusProProfile{T}(p::ClusProProfile) where {T} =
    ClusProProfile{T}(p.key, p.label, T(p.w_rep), T(p.w_attr), T(p.w_elec), T(p.w_dars))

function Base.show(io::IO, p::ClusProProfile{T}) where {T}
    print(io, "ClusProProfile{", T, "}(:", p.key, " \"", p.label, "\" ",
          "w_rep=", p.w_rep, " w_attr=", p.w_attr,
          " w_elec=", p.w_elec, " w_dars=", p.w_dars, ")")
end

"""
    shape_ratio(p) -> T

`w_rep / |w_attr|`: the repulsion-to-attraction magnitude ratio, which is the
only part of a 2017 profile that has a JuPIPER counterpart *inside* a grid
rather than beside it. It is what [`assumed_piper_weights`](@ref) hands to
`shape_receptor_grid!` as `w1`.

Across the seven published sets this takes exactly two values, **1.0** and
**2.5** — so five of the seven modes share one shape field. That is the fact
[`retarget_spectra!`](@ref) exploits, and `test_scoring.jl` pins it so a
transcription change cannot quietly invalidate the optimisation.

Compare 2006's published 2.1 and the 7.0 JuPIPER needed to dock 1GGR at all. The newest
parameterisation is the *least* repulsive of the three, which is only coherent if the
modern balance lives in the grids rather than the weights — which p.4 states outright.
"""
shape_ratio(p::ClusProProfile{T}) where {T} = p.w_rep / abs(p.w_attr)

"""
    load_cluspro_profiles(path) -> Vector{ClusProProfile{Float64}}

Read the seven 2017 coefficient sets from the provenance-headed TSV in `data/`.

The header on that file is long on purpose — it is where the four-term/three-term
mismatch is stated in full.

Allocates; setup-time. Throws on a duplicate key, a wrong field count, or a
non-negative `w_attr` (which would mean the transcription lost a sign, and
every energy computed through it would be repulsive where it should attract).
"""
function load_cluspro_profiles(path::AbstractString =
                                   joinpath(@__DIR__, "..", "data", "potentials",
                                            "cluspro2017_profiles.tsv"))
    out = ClusProProfile{Float64}[]
    seen = Set{Symbol}()
    lineno = 0

    open(path, "r") do io
    for line in eachline(io)
        lineno += 1
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s, '\t')
        length(f) == 6 || throw(ArgumentError(
            "$path:$lineno: expected 6 tab-separated fields, got $(length(f))"))
        strip(f[1]) == "key" && continue
        k = Symbol(strip(f[1]))
        k in seen && throw(ArgumentError("$path:$lineno: duplicate profile key :$k"))
        push!(seen, k)
        wr, wa = parse(Float64, f[2]), parse(Float64, f[3])
        we, wd = parse(Float64, f[4]), parse(Float64, f[5])
        wa < 0 || throw(ArgumentError(
            "$path:$lineno: w_attr must be negative (every published set is); " *
            "got $wa. A positive attractive weight would make contact cost " *
            "energy, which no transcription of Table 1 can mean."))
        push!(out, ClusProProfile{Float64}(k, String(strip(f[6])), wr, wa, we, wd))
    end
    end
    return out
end

"""
    cluspro_profile(key; profiles=load_cluspro_profiles) -> ClusProProfile

The named 2017 docking mode. Keys: `:balanced`, `:electrostatic`,
`:hydrophobic`, `:vdw_elec`, `:others1`, `:others2`, `:others3`.

Reloads the file on each call unless `profiles` is passed. That is deliberate
for a setup-time function: it is called once per run, and a cached global would
be one more piece of state to get wrong.
"""
function cluspro_profile(key::Symbol;
                         profiles::AbstractVector{<:ClusProProfile} =
                             load_cluspro_profiles())
    i = findfirst(p -> p.key === key, profiles)
    i === nothing && throw(ArgumentError(
        "unknown ClusPro profile :$key; known: " *
        join((":" * String(p.key) for p in profiles), ", ")))
    return profiles[i]
end

"""
    assumed_piper_weights(p) -> PiperWeights

Convert a 2017 four-weight profile into JuPIPER's three-weight form.

**The name says `assumed` because the result is [ASSUMED], and it is the one
word in this file doing real work.** A conversion constructor
`PiperWeights(p)` would have been the natural Julian spelling and is
deliberately not provided: it would let a caller move between two
parameterisations without ever typing anything that records the assumption.

## The arithmetic, which is exact

Divide the 2017 energy through by `|w_attr|`, which is positive:

    E_2017 / |w_attr| = (w_rep/|w_attr|)*E_rep - E_attr
                        + (w_elec/|w_attr|)*E_elec + (w_dars/|w_attr|)*E_DARS

The right-hand side is now in our shape (`E_attr` at weight −1, i.e. `−c` in
the field), so

    w1 = w_rep  / |w_attr|      (inside the shape field)
    w2 = w_elec / |w_attr|
    w3 = w_dars / |w_attr|

Dividing an energy by a positive constant cannot reorder poses, so **the
conversion is rank-preserving within a profile** — the returned weights select
exactly the same best pose the 2017 form would, on the same grids. That half is
a theorem and `test_scoring.jl` checks it on a landscape rather than trusting
it. Energies are *scaled*, so absolute values from two different profiles are
not comparable, and neither are ours to the reference's.

## What it assumes, which is false in one known respect

That our `E_rep` and `E_attr` grids carry the same relative normalisation as the
reference's. 2017 p.4 says theirs are "scaled to the van der Waals energy"; ours are
counts of atoms (PIPER preprint p.10). Under 2017's balanced set this returns `w1 =
1.0`, against 2006's published 2.1 and the 7.0 JuPIPER needed to dock 1GGR at all —
three ratios spanning 7×, of which ours is the largest and the newest reference's the
smallest.
"""
function assumed_piper_weights(p::ClusProProfile{T}) where {T}
    a = abs(p.w_attr)
    a > 0 || throw(ArgumentError(
        "w_attr = 0 leaves no attractive channel to normalise against; the " *
        "2006 form pins that weight at 1 and cannot represent the profile"))
    return PiperWeights{T}(p.w_rep / a, p.w_elec / a, p.w_dars / a)
end


"""
    rescale_terms!(rs, factors) -> rs

Multiply receptor spectrum `p` by `factors[p]`, in place.

Mutates `rs`. **Allocates nothing** after warm-up, on either backend: each term
is a broadcast into an array that already exists, and factors equal to 1 skip
their pass entirely rather than writing an array back unchanged.

This is the whole cost of changing a per-term scoring weight. The weights are
folded into the spectra at setup time by `spectra_from_grids`, so re-weighting
is `2 + P` scalar multiplies over data already in the right place — no
transform, no grid rebuild, and nothing touched inside the rotation loop.
"""
function rescale_terms!(rs::ReceptorSpectra{T}, factors::AbstractVector) where {T}
    length(factors) == nterms(rs) || throw(DimensionMismatch(
        "need one factor per term: $(nterms(rs)) terms, $(length(factors)) factors"))
    @inbounds for p in eachindex(rs.data)
        f = T(factors[p])
        f == one(T) && continue
        rs.data[p] .*= f
    end
    return rs
end

"""
    term_ratio!(dst, pot, from, to; P) -> dst

Fill `dst` with the per-term factor that turns receptor spectra weighted for `from` into
spectra weighted for `to`: `receptor_weights(pot, to)./ receptor_weights(pot, from)`, in
`build_termweights`' term order.

Mutates `dst`, which must hold `2 + P` elements. Allocates nothing.

**Throws when `from.w1 != to.w1`, and that refusal is the point.** `w1` multiplies the
*repulsive count inside the shape field* (`R = −c + w1 r`), not the field as a whole, so
no scalar applied to term 1 can change it. A version of this function that silently
rescaled term 1 anyway would return a wrong answer that looks entirely reasonable — the
failure mode this project keeps meeting. Changing `w1` costs one grid rebuild and one
r2c transform; see [`retarget_spectra!`](@ref), which names that path in its error
message.

`from.w2` or `from.w3` equal to zero is refused for the ordinary reason: a term
scaled to zero has lost its content, and nothing can be recovered from it by
multiplication. Rebuild the spectra instead. (The 2017 `vdw_elec` and `others3`
modes both set `w_dars = 0`, so this is reachable from published data, not
hypothetical.)
"""
function term_ratio!(dst::AbstractVector{T}, pot::Potential{T},
                     from::PiperWeights{T}, to::PiperWeights{T};
                     P::Integer) where {T}
    length(dst) == 2 + P || throw(DimensionMismatch(
        "dst must hold $(2 + P) factors for P = $P, got $(length(dst))"))
    from.w1 == to.w1 || throw(ArgumentError(
        "w1 differs ($(from.w1) -> $(to.w1)) and cannot be applied as a term " *
        "scale: w1 sits inside the shape field R = -c + w1*r, not in front of " *
        "it. Rebuild term 1 with `shape_receptor_grid!` and re-transform it " *
        "(one r2c), or rebuild the whole spectra with `piper_receptor_spectra`."))
    from.w2 == 0 && to.w2 != 0 && throw(ArgumentError(
        "the source weights zero the electrostatic term, so its spectrum holds " *
        "no content to rescale; rebuild the spectra instead"))
    from.w3 == 0 && to.w3 != 0 && throw(ArgumentError(
        "the source weights zero the pairwise term, so its spectra hold no " *
        "content to rescale; rebuild the spectra instead"))
    dst[1] = one(T)
    dst[2] = from.w2 == 0 ? one(T) : to.w2 / from.w2
    r3 = from.w3 == 0 ? one(T) : to.w3 / from.w3
    @inbounds for p in 1:P
        dst[2+p] = r3
    end
    return dst
end

"""
    retarget_spectra!(rs, buf, pot, from, to; P=nterms(rs)-2) -> rs

Re-weight an existing `ReceptorSpectra` from one scoring profile to another, in
place and without allocating.

Mutates `rs` and `buf`; `buf` must hold `2 + P` elements and exists so the
caller can hoist it. Composition of [`term_ratio!`](@ref) and
[`rescale_terms!`](@ref), and it inherits both refusals — in particular it will
not move between profiles with different `w1`, because that is a change to a
grid rather than to a weight.

## Why this is cheap, and when it is not

The scoring weights live in the receptor spectra, folded in at setup time.
Re-weighting is therefore `2 + P` scalar multiplies over arrays that already
exist: no FFT, no grid rebuild, no allocation, and nothing whatever inside the
rotation loop. It costs one pass over ~8 MiB of spectra against the ~70,000
rotations a run spends afterwards.

The exception is `w1`. Five of the seven published 2017 modes share
`shape_ratio == 1.0` and the remaining two share 2.5 (see
[`shape_ratio`](@ref)), so *most* mode switches are pure rescales and the two
that are not need one grid rebuild and one r2c transform — still setup-time,
still far below one rotation's share of a full run, but not free and not
silent.
"""
function retarget_spectra!(rs::ReceptorSpectra{T}, buf::AbstractVector{T},
                           pot::Potential{T}, from::PiperWeights{T},
                           to::PiperWeights{T};
                           P::Integer = nterms(rs) - 2) where {T}
    nterms(rs) == 2 + P || throw(DimensionMismatch(
        "spectra hold $(nterms(rs)) terms, expected $(2 + P) for P = $P"))
    term_ratio!(buf, pot, from, to; P = P)
    rescale_terms!(rs, buf)
    return rs
end

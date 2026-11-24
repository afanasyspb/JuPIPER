"""
    SpatialMask{T}

All-ones is the unmasked field exactly, so [`no_mask`](@ref) is a true identity
rather than an approximation of one.

Both channels are carried because the protocol has two independent biases and
they are not each other's inverse: repulsion *adds* a repulsive component, and
attraction *adds* an attractive one. Expressing both as one signed number would
make "mask this residue" and "favour this residue" the same axis, which they
are not -- a residue can be both known-not-interface and, say, a hydrophobic
patch, and the two statements compose.
"""
struct SpatialMask{T<:AbstractFloat}
    rep::Vector{T}
    attr::Vector{T}
end

Base.length(m::SpatialMask) = length(m.rep)

function Base.show(io::IO, m::SpatialMask{T}) where {T}
    nr = count(!=(one(T)), m.rep)
    na = count(!=(one(T)), m.attr)
    print(io, "SpatialMask{", T, "}(", length(m), " atoms, ", nr,
          " repulsion-biased, ", na, " attraction-biased)")
end

"""
    no_mask(T, natoms) -> SpatialMask{T}

The identity mask. Allocates two vectors; setup-time.
"""
no_mask(::Type{T}, n::Integer) where {T<:AbstractFloat} =
    SpatialMask{T}(ones(T, n), ones(T, n))

"""
    is_identity(mask) -> Bool

Whether the mask multiplies nothing. Used to skip the per-atom multiply, and
asserted by the tests that pin masking as a zero-cost-per-rotation change.
"""
is_identity(m::SpatialMask{T}) where {T} =
    all(==(one(T)), m.rep) && all(==(one(T)), m.attr)

"""
    mask_crossover(w_rep; radius = 1.7, D = 6, dpad = 2) -> Float64

The repulsion multiplier at which a masked atom stops being net attractive.
**A lower bound on any usable mask strength, derived rather than fitted.**

At uniform ligand density an atom contributes `+rep * w_rep` over the core
`r < d` (`d = radius + dpad`) and `-attr` over the shell `d < r < D`. The two
volumes are in the ratio `d^3 : (D^3 - d^3)`, so the atom's total contribution
is repulsive only when

    rep * w_rep > attr * ((D/d)^3 - 1)

That failure is silent -- the run completes, the energies are finite, and the poses are
wrong.

`radius` is the atom's van der Waals radius; the default is carbon's, which is
the modal heavy atom. Diagnostic, allocation-free, and not called in any loop.
"""
mask_crossover(w_rep::Real; radius::Real = 1.7, D::Real = 6, dpad::Real = 2) =
    ((D / (radius + dpad))^3 - 1) / w_rep

"""
    residue_mask(at, residues) -> BitVector

Per-atom flag for "this atom belongs to one of `residues`", each of which is a
`(chain, resnum)` pair. Allocates; setup-time.

Selection is by residue rather than by atom because that is the unit both the
protocol's interfaces use -- a named residue list and an uploaded PDB of whole
residues (2017 p.7). An atom-level mask is still expressible: build the
`BitVector` directly and hand it to [`spatial_mask!`](@ref).
"""
function residue_mask(at::AtomTable, residues)
    want = Set{Tuple{String,Int32}}()
    for (c, r) in residues
        push!(want, (String(c), Int32(r)))
    end
    sel = falses(length(at))
    @inbounds for i in eachindex(at.coords)
        (at.chain[i], at.resnum[i]) in want && (sel[i] = true)
    end
    return sel
end

"""
    spatial_mask(at, selected; repulsion, attraction) -> SpatialMask
    spatial_mask(T, natoms, selected; repulsion, attraction) -> SpatialMask

Build a mask that multiplies the shape field's repulsive channel by
`repulsion`, and its attractive channel by `attraction`, on the atoms flagged
in `selected`. Every other atom keeps 1.

**Neither strength has a default**. Pass `repulsion` above
[`mask_crossover`](@ref)`(w_rep)` for a mask that actually repels; the constructor
warns, once, if it is not.

Allocates; setup-time.
"""
function spatial_mask(::Type{T}, n::Integer, selected::AbstractVector{Bool};
                      repulsion::Real = 1, attraction::Real = 1) where {T<:AbstractFloat}
    m = no_mask(T, n)
    return spatial_mask!(m, selected; repulsion = repulsion, attraction = attraction)
end

spatial_mask(at::AtomTable{T}, selected::AbstractVector{Bool};
             repulsion::Real = 1, attraction::Real = 1) where {T} =
    spatial_mask(T, length(at), selected; repulsion = repulsion, attraction = attraction)

"""
    spatial_mask!(mask, selected; repulsion, attraction) -> mask

Apply one bias to an existing [`SpatialMask`](@ref). Mutates `mask`.

Biases **compose by multiplication**, so layering two selections that overlap
squares the multiplier on the overlap rather than silently taking the last one
written. That is the arithmetic a caller building "mask the non-CDR residues,
then additionally mask a known crystallisation artefact" wants, and it is the
reason this is a separate mutating entry point rather than a keyword on the
constructor.
"""
function spatial_mask!(m::SpatialMask{T}, selected::AbstractVector{Bool};
                       repulsion::Real = 1, attraction::Real = 1) where {T}
    length(selected) == length(m) ||
        throw(DimensionMismatch("selection must be one flag per atom " *
                                "($(length(m))), got $(length(selected))"))
    repulsion >= 0 || throw(ArgumentError("repulsion must be non-negative, got $repulsion"))
    attraction >= 0 ||
        throw(ArgumentError("attraction must be non-negative, got $attraction"))
    isfinite(repulsion) && isfinite(attraction) ||
        throw(ArgumentError("mask strengths must be finite: an infinite weight " *
                            "on a grid that is then Fourier-transformed is not a " *
                            "mask, it is a NaN factory (2017 p.7 says 'adding a " *
                            "repulsive force component', not 'infinite')"))
    r = T(repulsion); a = T(attraction)
    @inbounds for i in eachindex(selected)
        selected[i] || continue
        m.rep[i] *= r
        m.attr[i] *= a
    end
    return m
end

"""
    masked_residue_count(at, mask) -> Int

Number of distinct residues carrying a non-identity bias. Diagnostic: a mask
built from the wrong chain identifier selects nothing, and a count of zero is
the only cheap way to notice before a docking run reports no difference and
gets read as "masking does not help". Allocates.
"""
function masked_residue_count(at::AtomTable, m::SpatialMask{T}) where {T}
    length(at) == length(m) ||
        throw(DimensionMismatch("mask is for $(length(m)) atoms, table has $(length(at))"))
    seen = Set{Tuple{String,Int32}}()
    @inbounds for i in eachindex(at.coords)
        (m.rep[i] == one(T) && m.attr[i] == one(T)) && continue
        push!(seen, (at.chain[i], at.resnum[i]))
    end
    return length(seen)
end

"""
    RotationSet{T}

A sampled set of rigid-body rotations, in both quaternion and matrix form.

`quats` drives the geometry (distances, covering radius); `rots` is what the
docking pipeline consumes. Both are held because converting back and forth is
lossy busywork and the set is built once per run.

For a set built by [`rotations_hopf`](@ref), `ndir * nspin == length(quats)` and
the layout is **fibre-contiguous**: indices `(i-1)*nspin+1 : i*nspin` all share
base direction `i`, differing only by a twist about that direction. This is the
memory layout any direction-level pruning depends on; see [`fiber`](@ref).
For unstructured sets `ndir == nspin == 0`.
"""
struct RotationSet{T<:AbstractFloat}
    quats::Vector{SVector{4,T}}
    rots::Vector{SMatrix{3,3,T,9}}
    ndir::Int
    nspin::Int
    kind::Symbol
end

Base.length(rs::RotationSet) = length(rs.quats)
Base.getindex(rs::RotationSet, i::Integer) = rs.rots[i]
Base.eltype(::RotationSet{T}) where {T} = T

"""
    fiber(rs::RotationSet, i) -> UnitRange

Index range of the rotations sharing base direction `i`. Errors on an
unstructured set rather than inventing a grouping.
"""
function fiber(rs::RotationSet, i::Integer)
    rs.nspin > 0 || throw(ArgumentError("$(rs.kind) set has no fibre structure; use rotations_hopf"))
    1 <= i <= rs.ndir || throw(BoundsError(rs, i))
    return ((i - 1) * rs.nspin + 1):(i * rs.nspin)
end

"""
    rotmatrix(q::SVector{4}) -> SMatrix{3,3}

Unit quaternion `(w,x,y,z)` to rotation matrix. Assumes `q` is normalised;
the generators below guarantee it by construction.
"""
@inline function rotmatrix(q::SVector{4,T}) where {T}
    w, x, y, z = q[1], q[2], q[3], q[4]
    tx, ty, tz = 2x, 2y, 2z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return SMatrix{3,3,T,9}(
        one(T) - (tyy + tzz), txy + twz,            txz - twy,
        txy - twz,            one(T) - (txx + tzz), tyz + twx,
        txz + twy,            tyz - twx,            one(T) - (txx + tyy))
end


const SF_PHI = sqrt(2.0)
const SF_PSI = 1.533751168755204288118041

"""
    rotations_fibonacci(T, n) -> RotationSet{T}

`n` rotations from a super-Fibonacci spiral on S^3.

A single interleaved spiral with two incommensurate frequencies. Unlike a
product grid it has no preferred axis, which is why it achieves lower dispersion
per point than the Hopf construction -- at the cost of having no exploitable
structure.

Allocates (setup-time function, called once per docking run).

!!! note "Generated in Float64 regardless of `T`"
    The spiral phase is `2*pi*i/phi`, which reaches ~3e5 radians at `i = 7e4`.
    In Float32 the spacing of representable numbers there is ~0.03 rad = 1.8
    degrees, so the phase -- and therefore the whole set -- would be quantised
    far more coarsely than the ~5 degree grid it is trying to build. Generation
    is done in Float64 and narrowed at the end. `test_rotations.jl` pins this.
"""
function rotations_fibonacci(::Type{T}, n::Integer) where {T<:AbstractFloat}
    n > 0 || throw(ArgumentError("n must be positive"))
    quats = Vector{SVector{4,T}}(undef, n)
    @inbounds for i in 0:(n - 1)
        s = i + 0.5
        t = s / n
        d = 2pi * s
        r = sqrt(t)
        R = sqrt(1 - t)

        a = rem(d / SF_PHI, 2pi, RoundNearest)
        b = rem(d / SF_PSI, 2pi, RoundNearest)
        sa, ca = sincos(a)
        sb, cb = sincos(b)
        quats[i + 1] = SVector{4,T}(T(r * sa), T(r * ca), T(R * sb), T(R * cb))
    end
    return RotationSet{T}(quats, map(rotmatrix, quats), 0, 0, :fibonacci)
end

rotations_fibonacci(n::Integer) = rotations_fibonacci(Float64, n)


"""
    rotations_hopf(T, ndir, nspin) -> RotationSet{T}

`ndir * nspin` rotations from the Hopf fibration: `ndir` base directions on S^2,
each spun `nspin` times about itself.

In Hopf coordinates the Haar measure factorises,

    dV_SO(3)  =  dA_S2  x  d psi

so a uniform S^2 set crossed with a uniform twist grid is uniform on SO(3) --
no Jacobian correction. That is the property that makes this construction worth
having. Base directions come from a spherical Fibonacci spiral.

# The fibre property, which is the entire point

Within one fibre, `R * [1,0,0]` is **constant**: every member points the ligand's
own x-axis in the same lab direction, and consecutive members differ by a twist
of `2*pi/nspin` about that axis. Formally the fibre is a right coset
`R * Rx(psi)`, i.e. a body-frame spin.

This is not free -- the textbook Hopf parametrisation generates the *left*
coset `Rx(psi) * R`, whose invariant is `R' * xhat` and which groups together
poses with no physical resemblance (the whole ligand tumbling about a fixed lab
axis). We emit the conjugate quaternion so the grouping is the physically
meaningful one. Inversion is an isometry of SO(3), so every quality metric is
unchanged by this choice; only the interpretation of a fibre changes.

Output is fibre-contiguous; see [`fiber`](@ref).

Allocates (setup-time). Generated in Float64 for the reason given in
[`rotations_fibonacci`](@ref).
"""
function rotations_hopf(::Type{T}, ndir::Integer, nspin::Integer) where {T<:AbstractFloat}
    (ndir > 0 && nspin > 0) || throw(ArgumentError("ndir and nspin must be positive"))
    n = ndir * nspin
    quats = Vector{SVector{4,T}}(undef, n)
    golden = (1 + sqrt(5.0)) / 2
    k = 0
    @inbounds for i in 0:(ndir - 1)

        z = 1 - (2i + 1) / ndir
        theta = acos(clamp(z, -1.0, 1.0))
        phi = rem(2pi * i / golden, 2pi, RoundNearest)
        sth, cth = sincos(theta / 2)
        for j in 0:(nspin - 1)
            psi = 2pi * j / nspin
            spsi, cpsi = sincos(psi / 2)
            sph, cph = sincos(phi + psi / 2)
            k += 1

            quats[k] = SVector{4,T}(T(cth * cpsi), T(-cth * spsi),
                                    T(-sth * cph), T(-sth * sph))
        end
    end
    return RotationSet{T}(quats, map(rotmatrix, quats), Int(ndir), Int(nspin), :hopf)
end

rotations_hopf(ndir::Integer, nspin::Integer) = rotations_hopf(Float64, ndir, nspin)

"""
    hopf_split(n) -> (ndir, nspin)

Split `n` rotations into base directions and twists so that the grid is
*isotropic* -- the spacing along a fibre matches the spacing between fibres.

A Hopf set is a product grid, so it has two independent spacings and nothing
forces them to agree. Measurement (`bench/rotations.jl`) shows that in the SO(3)
metric

    fibre-neighbour distance   = 2*pi / nspin      (exact)
    direction-neighbour distance ~ sqrt(4*pi / ndir)  (the S^2 spacing)

Setting them equal gives `nspin = sqrt(pi * ndir)`, and with `n = ndir * nspin`,

    nspin = cbrt(pi * n)

For `n = 4096` this predicts `nspin = 23.4`. Measured efficiency forms a plateau
over `nspin` in 20..24 (0.723-0.731, individually within the 3.4% measurement
spread, so the exact optimum is *not* resolved) with the prediction falling
inside it. What is resolved, because the effects are an order of magnitude
larger than the spread, is the cost of getting it wrong: efficiency falls to
0.414 at `nspin = 8` and 0.212 at `nspin = 4`.

The failure is asymmetric and the intuitive direction is the wrong one. Reaching
for "many directions, a few spins each" -- 5000 x 14 at production scale --
measures 12.87 deg covering radius against 4.91 deg for the balanced split: 2.6x
worse, and worse than *random* sampling of the same size.

The returned `ndir * nspin` is the largest product not exceeding `n`, so a
caller asking for 70,000 may receive slightly fewer. Ask for the count you get,
do not pad -- `dock!` refuses a ragged batch by design.
"""
function hopf_split(n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    nspin = max(1, round(Int, cbrt(pi * n)))
    ndir = max(1, n ÷ nspin)
    return (ndir, nspin)
end

"""
    rotations_hopf_balanced(T, n) -> RotationSet{T}

`rotations_hopf` at the isotropic split from [`hopf_split`](@ref). The set has
`ndir * nspin <= n` rotations; check `length` rather than assuming `n`.
"""
function rotations_hopf_balanced(::Type{T}, n::Integer) where {T<:AbstractFloat}
    ndir, nspin = hopf_split(n)
    return rotations_hopf(T, ndir, nspin)
end

rotations_hopf_balanced(n::Integer) = rotations_hopf_balanced(Float64, n)

"""
    rotations_random(T, n; rng) -> RotationSet{T}

`n` Haar-uniform random rotations. Not for production docking -- it is the
control the structured sets are measured against, and the sampler used by
[`covering_radius`](@ref).

Uses the normalised-Gaussian construction: a 4-D standard normal is spherically
symmetric, so normalising it is uniform on S^3, hence Haar-uniform on SO(3).
"""
function rotations_random(::Type{T}, n::Integer;
                          rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    quats = Vector{SVector{4,T}}(undef, n)
    @inbounds for i in 1:n
        v = SVector{4,Float64}(Random.randn(rng), Random.randn(rng),
                               Random.randn(rng), Random.randn(rng))
        quats[i] = SVector{4,T}(v ./ norm(v))
    end
    return RotationSet{T}(quats, map(rotmatrix, quats), 0, 0, :random)
end

rotations_random(n::Integer; kw...) = rotations_random(Float64, n; kw...)


"""
    rotangle(q1, q2) -> Float64

Geodesic distance on SO(3) in radians: the angle of the rotation carrying one
to the other, in `[0, pi]`. Note the `abs` -- `q` and `-q` are the same
rotation.
"""
@inline function rotangle(q1::SVector{4}, q2::SVector{4})
    c = abs(Float64(q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3] + q1[4] * q2[4]))
    return 2 * acos(min(1.0, c))
end

"""
    covering_bound(n) -> Float64

Smallest covering radius any `n`-point set on SO(3) could have, in radians.

The normalised Haar measure of a ball of angular radius `theta` is
`(theta - sin(theta))/pi`. If `n` such balls are to cover the group they must
between them have total measure at least 1, giving

    theta - sin(theta) = pi / n

which is solved here by Newton from the small-angle start `(6 pi / n)^(1/3)`.
This is a packing-style bound and is not achievable; real sets land 1.2-1.6x
above it. Its use is as a denominator -- see [`covering_efficiency`](@ref) --
so that grids of different sizes can be compared on one scale.
"""
function covering_bound(n::Integer)
    n > 0 || throw(ArgumentError("n must be positive"))
    target = pi / n
    theta = cbrt(6 * target)
    for _ in 1:60
        f = theta - sin(theta) - target
        fp = 1 - cos(theta)
        fp <= 0 && break
        step = f / fp
        theta -= step
        abs(step) < 1e-15 && break
    end
    return clamp(theta, 0.0, pi)
end

"""
    covering_radius(rs; nsamples=20_000, nrepeat=5, rng) -> Float64

Monte-Carlo estimate of the covering radius in radians: the largest angular
distance from an arbitrary rotation to the nearest member of `rs`.

This is the metric that decides whether a rotation set is fit for docking. It
bounds how far the best sampled orientation can be from the true one, so it
translates directly into a floor on achievable interface RMSD -- no scoring
function, however good, recovers an orientation the grid never visited.

# The estimator is biased, and the bias is not small

A random probe rarely lands in the deepest hole, so a single batch
**underestimates**. Measured on a 4080-point Hopf set, the estimate of covering
*efficiency* falls monotonically as sampling improves:

    nsamples    2000     5000    20000    80000
    efficiency  0.769    0.748    0.722    0.721

and at fixed `nsamples = 20000` the seed-to-seed spread is 7.5% for a Hopf set
(2.0% for super-Fibonacci -- a product grid has fewer, more localised holes, so
they are harder to hit at random). A comparison between two constructions is
therefore only meaningful if the gap exceeds that spread.

The estimate improves monotonically with `nsamples` and nothing else: the
statistic is a maximum, so splitting probes into batches and taking the largest
batch maximum is exactly the same number as one batch of all of them. Spend the
budget on `nsamples`, and use [`covering_spread`](@ref) when you need to know
how much uncertainty is left. Cost is `O(nsamples * length(rs))`.
"""
function covering_radius(rs::RotationSet; nsamples::Integer = 20_000,
                         rng::Random.AbstractRNG = Random.default_rng())
    qs = rs.quats
    n = length(qs)
    worst = 0.0
    @inbounds for _ in 1:nsamples
        v = SVector{4,Float64}(Random.randn(rng), Random.randn(rng),
                               Random.randn(rng), Random.randn(rng))
        q = v ./ norm(v)
        best = 0.0
        for i in 1:n
            p = qs[i]
            c = abs(q[1] * p[1] + q[2] * p[2] + q[3] * p[3] + q[4] * p[4])
            c > best && (best = c)
        end
        d = 2 * acos(min(1.0, best))
        d > worst && (worst = d)
    end
    return worst
end

"""
    covering_spread(rs; nsamples=20_000, nrepeat=5, seed=1) -> (lo, med, hi)

Covering radius estimated `nrepeat` times from independent probe batches, in
radians. The spread of these is the honest error bar on a single
[`covering_radius`](@ref) call, and the point estimate to quote is `hi` -- the
statistic is a maximum and converges from below, so the largest run is the best
of them, not an outlier to be trimmed.

Use this before claiming one construction beats another: measured gaps between
good constructions are of the same order as this spread.
"""
function covering_spread(rs::RotationSet; nsamples::Integer = 20_000,
                         nrepeat::Integer = 5, seed::Integer = 1)
    v = [covering_radius(rs; nsamples = nsamples, rng = Random.Xoshiro(seed + k))
         for k in 1:nrepeat]
    sort!(v)
    return (v[1], v[(nrepeat + 1) >> 1], v[end])
end

"""
    covering_efficiency(rs; kw...) -> Float64

`covering_bound(n) / covering_radius(rs)`, in `(0, 1]`. Dimensionless, so it
compares sets of different sizes. 1.0 would be a perfect (unachievable) cover;
Haar-random sampling scores far below the structured constructions.
"""
covering_efficiency(rs::RotationSet; kw...) =
    covering_bound(length(rs)) / covering_radius(rs; kw...)

"""
    nn_stats(rs) -> (min, median, max) nearest-neighbour angles, radians

Spread of the within-set nearest-neighbour distances. A set with uniform local
density has a tight spread; a product grid that clumps near a pole does not.

Complements [`covering_radius`](@ref): covering radius sees the largest hole,
this sees whether points are evenly spaced everywhere. `O(n^2)`, so it is a
diagnostic for a few thousand points, not for the production set.
"""
function nn_stats(rs::RotationSet)
    qs = rs.quats
    n = length(qs)
    n > 1 || throw(ArgumentError("need at least two rotations"))
    d = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        best = 0.0
        qi = qs[i]
        for j in 1:n
            i == j && continue
            p = qs[j]
            c = abs(qi[1] * p[1] + qi[2] * p[2] + qi[3] * p[3] + qi[4] * p[4])
            c > best && (best = c)
        end
        d[i] = 2 * acos(min(1.0, best))
    end
    sort!(d)
    return (d[1], d[(n + 1) >> 1], d[n])
end

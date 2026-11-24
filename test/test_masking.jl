using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
using JuPIPER: GridSpec, Structure, CPUBackend, workspace, spectra_from_grids,
               correlate!, unravel, Pose, pose_transform, shape_receptor_grid!,
               AtomTable, IngestReport


"A uniformly random rotation, from a normalised quaternion (column-major SMatrix)."
function rand_rot(rng, ::Type{T}) where {T}
    q = randn(rng, 4); q ./= norm(q)
    w, x, y, z = q
    return SMatrix{3,3,T}(1 - 2(y^2 + z^2), 2(x * y + z * w), 2(x * z - y * w),
                          2(x * y - z * w), 1 - 2(x^2 + z^2), 2(y * z + x * w),
                          2(x * z + y * w), 2(y * z - x * w), 1 - 2(x^2 + y^2))
end

"An AtomTable built by hand: no reader, no fixture file, no include ordering."
function tiny_table(::Type{T}) where {T}

    coords = [SVector{3,T}(0, 0, 0), SVector{3,T}(1, 0, 0),
              SVector{3,T}(0, 2, 0), SVector{3,T}(1, 2, 0),
              SVector{3,T}(0, 0, 3), SVector{3,T}(1, 0, 3)]
    n = length(coords)
    rep = IngestReport("<synthetic>", 1, 1, ["A", "B"], ["A", "B"], n, n,
                       0, 0, 0, Dict{String,Int}(), 0, Dict{String,Int}(),
                       0, 0, 3, 0, 0)
    return AtomTable{T}(coords, fill("C", n), ["CA", "CB", "CA", "CB", "CA", "CB"],
                        fill("ALA", n), Int32[10, 10, 11, 11, 10, 10],
                        fill(' ', n), ["A", "A", "A", "A", "B", "B"],
                        ones(T, n), zeros(T, n), falses(n), falses(n),
                        SVector{3,T}(0, 0, 0), false, rep)
end

"A small docking case with signed term weights, as in `test_fibre.jl`."
function mask_case(::Type{T}, n::Int, nterms::Int, natoms::Int; seed = 99) where {T}
    rng = MersenneTwister(seed)
    spec = GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-n * 0.6, -n * 0.6, -n * 0.6))
    y = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:natoms]
    y .-= (sum(y) / natoms,)
    W = Matrix{T}(undef, natoms, nterms)
    @views W[:, 1] .= one(T)
    for p in 2:nterms
        @views W[:, p] .= randn(rng, natoms)
    end
    grids = [T.(randn(rng, n, n, n)) for _ in 1:nterms]
    weights = T[1, 3.5, -2.25, 0.75][1:nterms]
    return (spec, Structure(y, W), grids, weights)
end

@testset "spatial masking" begin

@testset "the identity mask is bit-identical, not merely close" begin

    for T in (Float32, Float64)
        spec = GridSpec{T}((16, 16, 16), T(1.2),
                           SVector{3,T}(-9.6, -9.6, -9.6))
        rng = MersenneTwister(7)
        xs = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:20]
        rad = fill(T(1.7), 20)
        A = Array{T,3}(undef, spec.n...)
        B = Array{T,3}(undef, spec.n...)
        shape_receptor_grid!(A, xs, rad, spec; w_rep = 2.1)
        m = no_mask(T, 20)
        @test is_identity(m)
        shape_receptor_grid!(B, xs, rad, spec; w_rep = 2.1,
                             rep_scale = m.rep, attr_scale = m.attr)
        @test A == B
    end
end

@testset "mask_crossover reproduces the crossover constant" begin

    @test mask_crossover(1.0) ≈ (6 / 3.7)^3 - 1
    @test mask_crossover(1.0) ≈ 3.264 atol = 1e-3

    @test mask_crossover(2.1) ≈ ((6 / 3.7)^3 - 1) / 2.1
    @test mask_crossover(7.0) ≈ ((6 / 3.7)^3 - 1) / 7.0
    @test mask_crossover(2.1) > 1
    @test mask_crossover(7.0) < 1
end

@testset "the crossover is where the discrete field actually flips sign" begin

    T = Float64
    nn, h = 40, T(0.4)
    spec = GridSpec{T}((nn, nn, nn), h,
                       SVector{3,T}(-nn * h / 2, -nn * h / 2, -nn * h / 2))
    xs = [SVector{3,T}(0, 0, 0)]
    rad = [T(1.7)]
    R = Array{T,3}(undef, spec.n...)
    w_rep = 2.1

    shape_receptor_grid!(R, xs, rad, spec; w_rep = 1.0)
    ncore  = count(≈(1.0), R)
    nshell = count(≈(-1.0), R)
    @test ncore > 0 && nshell > 0
    discrete = nshell / (w_rep * ncore)
    @test discrete ≈ mask_crossover(w_rep) rtol = 0.05

    below = discrete * 0.8
    above = discrete * 1.25
    for (mult, wantpos) in ((below, false), (above, true))
        m = spatial_mask(T, 1, [true]; repulsion = mult)
        shape_receptor_grid!(R, xs, rad, spec; w_rep = w_rep,
                             rep_scale = m.rep, attr_scale = m.attr)
        s = sum(R)
        @test (s > 0) == wantpos
    end

    m1 = spatial_mask(T, 1, [true]; repulsion = 1)
    shape_receptor_grid!(R, xs, rad, spec; w_rep = w_rep,
                         rep_scale = m1.rep, attr_scale = m1.attr)
    @test sum(R) < 0
end

@testset "masking changes grid content, never grid shape" begin

    T = Float64
    spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    rng = MersenneTwister(11)
    xs = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:30]
    rad = fill(T(1.7), 30)
    A = Array{T,3}(undef, spec.n...); B = similar(A)
    shape_receptor_grid!(A, xs, rad, spec; w_rep = 2.1)
    sel = falses(30); sel[1:5] .= true
    m = spatial_mask(T, 30, sel; repulsion = 4.0)
    shape_receptor_grid!(B, xs, rad, spec; w_rep = 2.1,
                         rep_scale = m.rep, attr_scale = m.attr)
    @test size(A) == size(B)
    @test A != B
    @test sum(B) > sum(A)
end

@testset "biases compose by multiplication, and refuse what they cannot mean" begin
    T = Float64
    m = no_mask(T, 4)
    sel_a = [true, true, false, false]
    sel_b = [false, true, true, false]
    spatial_mask!(m, sel_a; repulsion = 2)
    spatial_mask!(m, sel_b; repulsion = 3)
    @test m.rep == T[2, 6, 3, 1]
    @test m.attr == ones(T, 4)
    @test !is_identity(m)

    ma = spatial_mask(T, 3, [true, false, true]; attraction = 2.5)
    @test ma.attr == T[2.5, 1, 2.5]
    @test ma.rep == ones(T, 3)

    @test_throws ArgumentError spatial_mask(T, 3, [true, false, false];
                                            repulsion = Inf)
    @test_throws ArgumentError spatial_mask(T, 3, [true, false, false];
                                            repulsion = -1)
    @test_throws DimensionMismatch spatial_mask!(no_mask(T, 3), [true, false])
end

@testset "residue selection is by chain AND number" begin
    T = Float64
    at = tiny_table(T)
    @test length(at) == 6
    sel = residue_mask(at, [("A", 10)])
    @test sel == [true, true, false, false, false, false]
    sel2 = residue_mask(at, [("A", 10), ("B", 10)])
    @test sel2 == [true, true, false, false, true, true]
    @test residue_mask(at, [("C", 10)]) == falses(6)

    m = spatial_mask(at, sel2; repulsion = 3.0)
    @test length(m) == 6
    @test masked_residue_count(at, m) == 2
    @test masked_residue_count(at, no_mask(T, 6)) == 0
    @test_throws DimensionMismatch masked_residue_count(at, no_mask(T, 5))
end

end


@testset "distance restraints" begin

"Feasible cells by brute force, THROUGH `pose_transform` -- the real convention."
function brute_feasible(set::RestraintSet{T}, rec, ligy, R, spec) where {T}
    out = Int32[]
    rots = [R]
    placed = similar(ligy)
    for c in 1:prod(spec.n)
        x, y, z = unravel(spec, c)
        p = Pose{T}(zero(T), Int32(1), Int16(x), Int16(y), Int16(z))
        Rm, t = pose_transform(p, rots, spec)
        for k in eachindex(ligy)
            placed[k] = Rm * ligy[k] + t
        end
        ngroup = 0
        for g in set.groups
            nsat = 0
            for r in g.members
                sat = false
                for i in r.rec, j in r.lig
                    d = norm(rec[i] - placed[j])
                    if r.dmin <= d <= r.dmax
                        sat = true; break
                    end
                end
                sat && (nsat += 1)
            end
            nsat > g.cutoff && (ngroup += 1)
        end
        ngroup > set.cutoff && push!(out, Int32(c))
    end
    return out
end

@testset "the feasible set equals brute force, exactly" begin

    T = Float64
    rng = MersenneTwister(5)
    spec = GridSpec{T}((14, 16, 12), T(1.2), SVector{3,T}(-8.4, -9.6, -7.2))
    rec = [SVector{3,T}(3randn(rng), 3randn(rng), 3randn(rng)) for _ in 1:6]
    ligy = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:5]
    ligy .-= (sum(ligy) / length(ligy),)

    rs = [restraint([1], [1], 0.0, 4.0; T = T),
          restraint([2, 3], [2], 1.5, 6.0; T = T),
          restraint([4], [3, 4, 5], 0.0, 5.0; T = T)]
    ws = restraint_workspace(T, spec)

    for (gcut, scut) in ((0, 0), (1, 0))
        set = restraint_set([restraint_group(rs; cutoff = gcut)]; cutoff = scut)
        for trial in 1:4
            R = trial == 1 ? SMatrix{3,3,T}(I) : rand_rot(rng, T)
            got = copy(feasible_translations!(ws, set, rec, ligy, R, spec))
            want = brute_feasible(set, rec, ligy, R, spec)
            @test got == want
        end
    end
end

@testset "a single pair lands where the convention says, not mirrored" begin

    T = Float64
    spec = GridSpec{T}((24, 24, 24), T(1.0), SVector{3,T}(-12, -12, -12))
    rec = [SVector{3,T}(6, 0, 0)]
    ligy = [SVector{3,T}(0, 0, 0)]
    set = restraint_set([restraint([1], [1], 0.0, 1.4; T = T)])
    ws = restraint_workspace(T, spec)
    got = copy(feasible_translations!(ws, set, rec, ligy, SMatrix{3,3,T}(I), spec))
    @test !isempty(got)
    @test got == brute_feasible(set, rec, ligy, SMatrix{3,3,T}(I), spec)

    rots = [SMatrix{3,3,T}(I)]
    for c in got
        x, y, z = unravel(spec, c)
        Rm, t = pose_transform(Pose{T}(zero(T), Int32(1), Int16(x), Int16(y), Int16(z)),
                               rots, spec)
        @test norm((Rm * ligy[1] + t) - rec[1]) <= 1.4 + 1e-12
    end

    mirrored = Set(Int32[])
    for c in got
        x, y, z = unravel(spec, c)
        push!(mirrored, Int32(1 + mod(-x, 24) + 24 * (mod(-y, 24) + 24 * mod(-z, 24))))
    end
    @test isempty(intersect(Set(got), mirrored))
end

@testset "group and set cutoffs are counts, and the paper's is strict" begin
    T = Float64
    spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    rec = [SVector{3,T}(0, 0, 0), SVector{3,T}(8, 0, 0)]
    ligy = [SVector{3,T}(0, 0, 0)]
    r1 = restraint([1], [1], 0.0, 3.0; T = T)
    r2 = restraint([2], [1], 0.0, 3.0; T = T)
    ws = restraint_workspace(T, spec)

    union_set = restraint_set([restraint_group([r1, r2]; cutoff = 0)])
    inter_set = restraint_set([restraint_group([r1, r2]; cutoff = 1)])
    R = SMatrix{3,3,T}(I)
    u = copy(feasible_translations!(ws, union_set, rec, ligy, R, spec))
    i = copy(feasible_translations!(ws, inter_set, rec, ligy, R, spec))
    @test !isempty(u)
    @test length(i) < length(u)
    @test issubset(Set(i), Set(u))
    @test i == brute_feasible(inter_set, rec, ligy, R, spec)

    @test isempty(i)

    two = restraint_set([restraint_group([r1]), restraint_group([r2])]; cutoff = 0)
    @test Set(copy(feasible_translations!(ws, two, rec, ligy, R, spec))) == Set(u)

    @test_throws ArgumentError restraint_group([r1, r2]; cutoff = 2)
    @test_throws ArgumentError restraint_set(RestraintGroup{T}[])
    @test_throws ArgumentError restraint(Int[], [1], 0.0, 3.0; T = T)
    @test_throws ArgumentError restraint([1], [1], 3.0, 3.0; T = T)
    @test_throws ArgumentError restraint([1], [1], -1.0, 3.0; T = T)
    @test nrestraints(union_set) == 2
end

@testset "the workspace is reusable: repeated calls do not accumulate" begin

    T = Float64
    rng = MersenneTwister(31)
    spec = GridSpec{T}((12, 12, 12), T(1.2), SVector{3,T}(-7.2, -7.2, -7.2))
    rec = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:4]
    ligy = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) for _ in 1:3]
    set = restraint_set([restraint([1, 2], [1], 0.0, 4.0; T = T),
                         restraint([3], [2, 3], 1.0, 5.0; T = T)]; cutoff = 0)
    ws = restraint_workspace(T, spec)
    R = SMatrix{3,3,T}(I)
    first_call = copy(feasible_translations!(ws, set, rec, ligy, R, spec))
    for _ in 1:5
        @test feasible_translations!(ws, set, rec, ligy, R, spec) == first_call
    end

    feasible_translations!(ws, set, rec, ligy, rand_rot(rng, T), spec)
    @test feasible_translations!(ws, set, rec, ligy, R, spec) == first_call
end

@testset "dock_restrained! is the exhaustive answer restricted to the mask" begin

    T = Float64
    spec, lig, grids, weights = mask_case(T, 16, 3, 12)
    rng = MersenneTwister(77)
    rec = [SVector{3,T}(3randn(rng), 3randn(rng), 3randn(rng)) for _ in 1:5]
    rots = [rand_rot(rng, T) for _ in 1:6]
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)

    set = restraint_set([restraint([1, 2], [1, 2], 0.0, 7.0; T = T),
                         restraint([3], [3, 4], 0.0, 6.0; T = T)]; cutoff = 0)
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, length(rots))
    poses, rep = dock_restrained!(poses, lig, rots, rsp, pool, set, rec)

    ws = workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)
    rw = restraint_workspace(T, spec)
    nskip = 0
    for (i, R) in enumerate(rots)
        cells = copy(feasible_translations!(rw, set, rec, lig.coords, R, spec))
        @test cells == brute_feasible(set, rec, lig.coords, R, spec)
        if isempty(cells)
            nskip += 1
            @test poses[i].energy == typemax(T)
            continue
        end
        E = correlate!(ws, lig, view(rots, i:i), rsp)
        best = typemax(T); bi = 0
        for c in cells
            if E[c] < best
                best = E[c]; bi = Int(c)
            end
        end
        @test poses[i].energy === best
        x, y, z = unravel(spec, bi)
        @test (poses[i].tx, poses[i].ty, poses[i].tz) == (Int16(x), Int16(y), Int16(z))
    end
    @test rep.nrot == length(rots)
    @test rep.nskipped == nskip
    @test 0.0 <= feasible_fraction(rep) <= 1.0
    @test skip_rate(rep) ≈ nskip / length(rots)
end

@testset "an unsatisfiable restraint skips every rotation's transforms" begin

    T = Float64
    spec, lig, grids, weights = mask_case(T, 16, 3, 10)
    rng = MersenneTwister(5150)
    rec = [SVector{3,T}(0, 0, 0)]
    rots = [rand_rot(rng, T) for _ in 1:4]
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)

    far = restraint_set([restraint([1], [1], 500.0, 600.0; T = T)])
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, length(rots))
    poses, rep = dock_restrained!(poses, lig, rots, rsp, pool, far, rec)
    @test rep.nskipped == length(rots)
    @test skip_rate(rep) == 1.0
    @test feasible_fraction(rep) == 0.0
    @test all(p -> p.energy == typemax(T), poses)

    everywhere = restraint_set([restraint([1], [1], 0.0, 1e6; T = T)])
    poses2 = Vector{Pose{T}}(undef, length(rots))
    poses2, rep2 = dock_restrained!(poses2, lig, rots, rsp, pool, everywhere, rec)
    @test rep2.nskipped == 0
    @test feasible_fraction(rep2) ≈ 1.0
    plain = Vector{Pose{T}}(undef, length(rots))
    dock!(plain, lig, rots, rsp, pool)
    for i in eachindex(rots)
        @test poses2[i].energy === plain[i].energy
        @test (poses2[i].tx, poses2[i].ty, poses2[i].tz) ===
              (plain[i].tx, plain[i].ty, plain[i].tz)
    end
end

@testset "the driver refuses a batched pool" begin
    T = Float64
    spec, lig, grids, weights = mask_case(T, 16, 3, 8)
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 2, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, 2)
    set = restraint_set([restraint([1], [1], 0.0, 5.0; T = T)])
    @test_throws ArgumentError dock_restrained!(poses, lig, [SMatrix{3,3,T}(I),
                                                            SMatrix{3,3,T}(I)],
                                                rsp, pool, set,
                                                [SVector{3,T}(0, 0, 0)])
end

end

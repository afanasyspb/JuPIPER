using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
using JuPIPER: GridSpec, Structure, CPUBackend, workspace, spectra_from_grids,
               correlate!, unravel, Pose, pose_transform, minimal_image,
               rotmatrix, scan_best!


using BioStructures


"A small symmetric-docking case: grid, ligand, receptor grids, term weights."
function sym_case(::Type{T}, n::Int, nterms::Int, natoms::Int; seed = 4242) where {T}
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

"Rotation matrix for a turn of `ang` radians about unit axis `n`, built the long
way round so it is an independent construction from `cyclic_symmetry`'s."
function rodrigues(n::SVector{3,T}, ang::Real) where {T}
    s, c = sincos(T(ang))
    K = SMatrix{3,3,T,9}(0, n[3], -n[2], -n[3], 0, n[1], n[2], -n[1], 0)
    return SMatrix{3,3,T,9}(LinearAlgebra.I) + s * K + (1 - c) * (K * K)
end

@testset "Cn multimer symmetry" begin


@testset "axis sets on S^2" begin
    for T in (Float32, Float64)
        a = symmetry_axes(T, 500)
        @test length(a) == 500
        @test all(x -> abs(norm(x) - 1) < 10 * eps(T) * 10, a)
        @test eltype(a) == SVector{3,T}
        h = symmetry_axes(T, 500; hemisphere = true)
        @test length(h) == 500
        @test all(x -> x[3] > 0, h)

        @test abs(sum(x -> x[3], a) / 500) < 0.01
        @test abs(sum(x -> x[3], h) / 500 - 0.5) < 0.01
    end
    @test_throws ArgumentError symmetry_axes(Float64, 0)
end

@testset "hopf_axes is the fibre invariant, read off the set" begin
    rs = rotations_hopf(Float64, 37, 11)
    ax = hopf_axes(rs)
    @test length(ax) == rs.ndir == 37
    xhat = SVector{3,Float64}(1, 0, 0)
    for i in 1:rs.ndir

        for j in fiber(rs, i)
            @test norm(rs.rots[j] * xhat - ax[i]) < 1e-12
        end
    end
    @test_throws ArgumentError hopf_axes(rotations_fibonacci(Float64, 16))
end


@testset "the rotation is a turn of exactly 2pi/N about its axis" begin
    for T in (Float32, Float64), N in (2, 3, 4, 6, 5, 7)
        ax = symmetry_axes(T, 13)
        sym = cyclic_symmetry(N, ax)
        @test length(sym) == 13
        @test sym.order == N
        tol = T === Float32 ? 1f-5 : 1e-13
        for (i, n) in enumerate(ax)
            R = sym.rots[i]
            @test norm(R * n - n) < tol
            u, ang = axis_angle(R)
            @test abs(rad2deg(ang) - 360 / N) < (T === Float32 ? 1f-2 : 1e-10)
            @test min(norm(u - n), norm(u + n)) < (T === Float32 ? 1f-3 : 1e-9)
            @test norm(R - rodrigues(n, 2pi / N)) < (T === Float32 ? 1f-5 : 1e-14)
            M = R
            for _ in 2:N; M = R * M; end
            @test norm(M - SMatrix{3,3,T,9}(LinearAlgebra.I)) < tol
        end
    end
end

@testset "C2 antipodal axes are the SAME matrix, bitwise -- and C3's are not" begin

    for T in (Float32, Float64)
        for n in symmetry_axes(T, 40)
            @test cyclic_symmetry(2, [n]).rots[1] == cyclic_symmetry(2, [-n]).rots[1]

            @test cyclic_symmetry(3, [n]).rots[1] != cyclic_symmetry(3, [-n]).rots[1]
        end
    end
end

@testset "the Cn closure condition: sum_k R^k t == N (n.t) n" begin

    rng = MersenneTwister(11)
    for N in (2, 3, 4, 6)
        for n in symmetry_axes(Float64, 7)
            R = cyclic_symmetry(N, [n]).rots[1]
            t = SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) * 20
            tperp = t - dot(t, n) * n
            s = zero(SVector{3,Float64}); v = tperp
            for _ in 1:N; s += v; v = R * v; end
            @test norm(s) < 1e-11
            a = 1.7
            s2 = zero(SVector{3,Float64}); v = tperp + a * n
            for _ in 1:N; s2 += v; v = R * v; end
            @test norm(s2 - N * a * n) < 1e-11
        end
    end
end

@testset "axis_angle survives the angle a real dimer sits at" begin

    n = normalize(SVector{3,Float64}(0.5080, -0.0990, -0.8556))
    for deg in (179.457, 179.99, 180.0, 90.0, 0.5)
        R = rodrigues(n, deg2rad(deg))
        u, ang = axis_angle(R)
        @test abs(rad2deg(ang) - deg) < 1e-9
        @test min(norm(u - n), norm(u + n)) < 1e-8
    end
    R = rodrigues(n, deg2rad(179.99))
    naive = SVector(R[3,2] - R[2,3], R[1,3] - R[3,1], R[2,1] - R[1,2])
    naive = naive / norm(naive)
    u, _ = axis_angle(R)
    @test min(norm(u - n), norm(u + n)) <= min(norm(naive - n), norm(naive + n))

    @test axis_angle(SMatrix{3,3,Float64,9}(LinearAlgebra.I))[2] == 0.0
end


@testset "symmetry_fit recovers the operation, and rise is origin-free" begin
    rng = MersenneTwister(5)
    P = [SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) * 9 for _ in 1:60]
    for N in (2, 3, 4, 6)
        n = normalize(SVector{3,Float64}(0.3, -0.7, 0.5))
        R = cyclic_symmetry(N, [n]).rots[1]
        tperp = SVector{3,Float64}(1.0, 2.0, 3.0)
        tperp -= dot(tperp, n) * n
        Q = [R * p + tperp for p in P]
        f = symmetry_fit(P, Q)
        @test f.order == N
        @test abs(f.angle - 360 / N) < 1e-8
        @test abs(f.rise) < 1e-9
        @test f.defect < 1e-8
        @test f.rmsd < 1e-9

        Qs = [q + 1.7 * n for q in Q]
        fs = symmetry_fit(P, Qs)
        @test abs(fs.rise - 1.7) < 1e-8

        sh = SVector{3,Float64}(31.0, -14.0, 7.0)
        fo = symmetry_fit([p + sh for p in P], [q + sh for q in Qs])
        @test abs(abs(fo.rise) - abs(fs.rise)) < 1e-6
        N > 2 && @test abs(fo.rise - fs.rise) < 1e-6
        @test fo.order == fs.order
    end

    f0 = symmetry_fit(P, [p + SVector{3,Float64}(1, 2, 3) for p in P])
    @test f0.order == 0
end


@testset "plane_cells matches brute force over every cell" begin

    for (nx, ny, nz) in ((16, 20, 12), (14, 14, 14), (10, 24, 18))
        spec = GridSpec{Float64}((nx, ny, nz), 1.2, SVector{3,Float64}(0, 0, 0))
        for a in symmetry_axes(Float64, 7), tol in (0.4, 0.6, 1.2, 2.0, 3.7)
            got = sort!(plane_cells(a, spec, tol))
            want = Int32[]
            for cz in 0:nz-1, cy in 0:ny-1, cx in 0:nx-1
                m = SVector{3,Float64}(minimal_image(cx, nx), minimal_image(cy, ny),
                                       minimal_image(cz, nz))
                spec.spacing * abs(dot(m, a)) <= tol &&
                    push!(want, Int32(1 + cx + nx * (cy + ny * cz)))
            end
            @test got == sort!(want)
            @test length(unique(got)) == length(got)
        end
    end
end

@testset "plane_cells: the origin is always in, and the slab is monotone" begin
    spec = GridSpec{Float64}((20, 16, 24), 1.2, SVector{3,Float64}(0, 0, 0))
    for a in symmetry_axes(Float64, 11)
        prev = Set{Int32}()
        for tol in (0.3, 0.6, 1.2, 2.4, 4.8)
            s = Set(plane_cells(a, spec, tol))
            @test Int32(1) in s
            @test issubset(prev, s)
            prev = s
        end
    end
end

@testset "sym_tol_floor is the spacing at which the slab stops undersampling" begin
    @test sym_tol_floor(1.2) == 0.6
    @test sym_tol_floor(1.0) == 0.5

    nx, ny, nz = 20, 16, 24
    spec = GridSpec{Float64}((nx, ny, nz), 1.2, SVector{3,Float64}(0, 0, 0))
    ez = SVector{3,Float64}(0, 0, 1)
    @test length(plane_cells(ez, spec, sym_tol_floor(spec.spacing))) == nx * ny
    @test length(plane_cells(ez, spec, 0.99 * sym_tol_floor(spec.spacing))) == nx * ny
    @test length(plane_cells(ez, spec, 1.0 * spec.spacing)) == 3 * nx * ny

    for a in symmetry_axes(Float64, 5)
        c1 = length(plane_cells(a, spec, 2.4))
        c2 = length(plane_cells(a, spec, 4.8))
        @test 1.8 < c2 / c1 < 2.2
    end
    @test CLUSPRO_SYM_TOL == 2.0
end


@testset "symmetric_search_floor: a reachable target is reached exactly" begin

    T = Float64
    spec = GridSpec{T}((24, 24, 24), T(1.2), SVector{3,T}(-14.4, -14.4, -14.4))
    ax = symmetry_axes(T, 9)
    sym = cyclic_symmetry(2, ax; tol = 2.0)
    rng = MersenneTwister(3)
    y = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:30]
    y .-= (sum(y) / 30,)
    k = 4
    cells = plane_cells(sym.axes[k], spec, sym.tol)
    lin = cells[length(cells) ÷ 3]
    cx, cy, cz = unravel(spec, Int(lin))
    m = SVector{3,T}(minimal_image(cx, spec.n[1]), minimal_image(cy, spec.n[2]),
                     minimal_image(cz, spec.n[3]))
    t = -spec.spacing * m
    z = [sym.rots[k] * v + t for v in y]
    f = symmetric_search_floor(sym, spec, y, z)
    @test f.rmsd < 1e-12
    @test f.rot == k
    @test f.cells == (Int(m[1]), Int(m[2]), Int(m[3]))
end

@testset "symmetric_search_floor equals brute force, and cannot beat the free one" begin
    T = Float64
    spec = GridSpec{T}((18, 18, 18), T(1.2), SVector{3,T}(-10.8, -10.8, -10.8))
    sym = cyclic_symmetry(3, symmetry_axes(T, 8); tol = 1.5)
    rng = MersenneTwister(21)
    y = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:25]
    y .-= (sum(y) / 25,)
    z = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:25]

    best = T(Inf)
    for r in eachindex(sym.rots), lin in plane_cells(sym.axes[r], spec, sym.tol)
        cx, cy, cz = unravel(spec, Int(lin))
        t = -spec.spacing * SVector{3,T}(minimal_image(cx, spec.n[1]),
                                         minimal_image(cy, spec.n[2]),
                                         minimal_image(cz, spec.n[3]))
        s = zero(T)
        for k in eachindex(y)
            d = sym.rots[r] * y[k] + t - z[k]
            s += dot(d, d)
        end
        best = min(best, sqrt(s / length(y)))
    end
    f = symmetric_search_floor(sym, spec, y, z)
    @test f.rmsd ≈ best atol=1e-12

    @test f.rmsd >= search_floor(sym.rots, spec, y, z).rmsd - 1e-12

    @test_throws ArgumentError symmetric_search_floor(sym, spec, y, z;
                                                      mask = falses(length(y)))
end


@testset "dock_symmetric! equals the exhaustive landscape over the same cells" begin

    for T in (Float32, Float64), (n, nterms) in ((16, 2), (20, 3))
        spec, lig, grids, weights = sym_case(T, n, nterms, 24)
        rsp = spectra_from_grids(CPUBackend(), grids, spec, weights;
                                 flags = FFTW.ESTIMATE)
        sym = cyclic_symmetry(2, symmetry_axes(T, 6); tol = 1.7)
        pool = [workspace(CPUBackend(), spec, nterms; batch = 1, flags = FFTW.ESTIMATE)]
        poses = Vector{Pose{T}}(undef, length(sym))
        dock_symmetric!(poses, lig, sym, rsp, pool; nrot_full = 6 * 7)
        ws = pool[1]
        for i in 1:length(sym)
            correlate!(ws, lig, view(sym.rots, i:i), rsp)
            E = ws.score
            best = typemax(T); bi = 0
            for c in plane_cells(sym.axes[i], spec, sym.tol)
                v = E[Int(c)]
                if v < best || (v == best && Int(c) < bi)
                    best = v; bi = Int(c)
                end
            end
            x, y, z = unravel(spec, bi)
            @test poses[i].energy === best
            @test (Int(poses[i].tx), Int(poses[i].ty), Int(poses[i].tz)) == (x, y, z)
            @test Int(poses[i].rot) == i
        end
    end
end

@testset "the reduction is order-independent when the tie-break decides" begin

    T = Float64
    n, nterms = 16, 1
    spec = GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    y = [SVector{3,T}(0, 0, 0)]
    W = ones(T, 1, nterms)
    lig = Structure(y, W)
    grids = [fill(T(2), n, n, n)]
    rsp = spectra_from_grids(CPUBackend(), grids, spec, T[1]; flags = FFTW.ESTIMATE)
    sym = cyclic_symmetry(2, symmetry_axes(T, 5); tol = 2.0)
    pool = [workspace(CPUBackend(), spec, nterms; batch = 1, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, length(sym))
    dock_symmetric!(poses, lig, sym, rsp, pool)
    for i in 1:length(sym)
        cells = plane_cells(sym.axes[i], spec, sym.tol)

        correlate!(pool[1], lig, view(sym.rots, i:i), rsp)
        E = pool[1].score
        mn = minimum(E[Int(c)] for c in cells)
        want = minimum(Int(c) for c in cells if E[Int(c)] == mn)
        x, yy, z = unravel(spec, want)
        @test (Int(poses[i].tx), Int(poses[i].ty), Int(poses[i].tz)) == (x, yy, z)
    end
end

@testset "the driver's refusals" begin
    T = Float64
    spec, lig, grids, weights = sym_case(T, 16, 2, 12)
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    sym = cyclic_symmetry(2, symmetry_axes(T, 4); tol = 1.5)
    pool1 = [workspace(CPUBackend(), spec, 2; batch = 1, flags = FFTW.ESTIMATE)]
    pool2 = [workspace(CPUBackend(), spec, 2; batch = 2, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, length(sym))
    @test_throws ArgumentError dock_symmetric!(poses, lig, sym, rsp, pool2)
    @test_throws ArgumentError dock_symmetric!(poses, lig, sym, rsp,
                                               typeof(pool1)(undef, 0))
    @test_throws DimensionMismatch dock_symmetric!(Vector{Pose{T}}(undef, 3), lig,
                                                   sym, rsp, pool1)

    @test_throws ArgumentError cyclic_symmetry(1, symmetry_axes(T, 4))
    @test_throws ArgumentError cyclic_symmetry(2, symmetry_axes(T, 4); tol = 0.0)
    @test_throws ArgumentError cyclic_symmetry(2, SVector{3,T}[])
end


@testset "the search-space reduction is exactly nspin, by construction" begin

    for nrot in (600, 4096, 70_000)
        ndir, nspin = hopf_split(nrot)
        rs = rotations_hopf_balanced(Float64, nrot)
        @test (rs.ndir, rs.nspin) == (ndir, nspin)
        sym = cyclic_symmetry(2, rs)
        @test length(sym) == rs.ndir
        @test length(rs) / length(sym) == rs.nspin
    end

    @test hopf_split(70_000) == (1166, 60)
    rs = rotations_hopf_balanced(Float64, 70_000)
    @test length(rs) == 69_960
    @test length(cyclic_symmetry(2, rs)) == 1166
    fibre_spacing_deg = 360 / rs.nspin
    dir_spacing_deg = rad2deg(sqrt(4pi / rs.ndir))
    @test abs(fibre_spacing_deg - dir_spacing_deg) / fibre_spacing_deg < 0.05
end

@testset "the report keeps the two reductions apart" begin
    T = Float64
    spec, lig, grids, weights = sym_case(T, 16, 2, 12)
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    sym = cyclic_symmetry(2, symmetry_axes(T, 9); tol = 1.5)
    pool = [workspace(CPUBackend(), spec, 2; batch = 1, flags = FFTW.ESTIMATE)]
    poses = Vector{Pose{T}}(undef, length(sym))
    _, rep = dock_symmetric!(poses, lig, sym, rsp, pool; nrot_full = 9 * 12)
    @test rep.naxes == 9
    @test rep.nrot_full == 9 * 12
    @test rotation_reduction(rep) == 12.0
    total = sum(length(plane_cells(sym.axes[i], spec, sym.tol)) for i in 1:9)
    @test rep.plane_total == total
    @test translation_reduction(rep) ≈ prod(spec.n) * 9 / total

    _, rep0 = dock_symmetric!(poses, lig, sym, rsp, pool)
    @test rotation_reduction(rep0) == 1.0
end

@testset "the constrained search really is cheaper (regression guard, not a measurement)" begin

    T = Float32
    ndir, nspin = 32, 8
    spec, lig, grids, weights = sym_case(T, 32, 3, 40)
    rsp = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    rs = rotations_hopf(T, ndir, nspin)
    sym = cyclic_symmetry(2, rs; tol = 2.0)
    pool = [workspace(CPUBackend(), spec, 3; batch = 1, flags = FFTW.ESTIMATE)]
    sp = Vector{Pose{T}}(undef, length(sym))
    fp = Vector{Pose{T}}(undef, length(rs))
    dock_symmetric!(sp, lig, sym, rsp, pool)
    dock!(view(fp, 1:1), lig, view(rs.rots, 1:1), rsp, pool)
    t_sym = @elapsed dock_symmetric!(sp, lig, sym, rsp, pool)
    t_full = @elapsed dock!(fp, lig, rs.rots, rsp, pool)
    @test t_full / t_sym > nspin / 4
end


@testset "4XAU is a closed C2 homodimer" begin
    path = joinpath(@__DIR__, "..", "data", "structures", "4XAU.pdb")
    if !isfile(path)
        @info """skipping the 4XAU testset: data/structures/ is not committed.
                 Fetch it with `bash scripts/fetch_structures.sh 4XAU`."""
        @test_skip false
    else
        A = read_atoms(path; T = Float64, chains = "A")
        B = read_atoms(path; T = Float64, chains = "B")
        ca(at) = Dict((at.resnum[i], at.inscode[i]) => i for i in 1:length(at)
                      if at.atomname[i] == "CA" && at.element[i] == "C")
        da, db = ca(A), ca(B)
        ks = sort!(collect(intersect(keys(da), keys(db))))
        @test length(ks) == 372
        sh = B.centroid - A.centroid
        P = [A.coords[da[k]] for k in ks]
        Q = [B.coords[db[k]] + sh for k in ks]
        f = symmetry_fit(P, Q)

        @test f.order == 2
        @test f.defect < 1.0
        @test abs(f.rise) < 0.1
        @test f.rmsd < 1.0

        @test abs(f.rise) < sym_tol_floor(1.2)
    end
end

end

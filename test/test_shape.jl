"""
Geometric ("lock and key") shape scoring -- PIPER's `E_shape` term, PDF p.10.

The load-bearing test is the last one: a one-term `ReceptorSpectra` built from an
explicitly-constructed grid must reproduce the direct correlation. That is the
same guarantee `test_correctness.jl` gives the general path, re-checked for the
`spectra_from_grids` seam, because that seam is how all future real physics
enters the pipeline.
"""

using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
const JP = JuPIPER

"""Direct O(cells * atoms) evaluation of the paper's definition. Slow, obvious."""
function shape_grid_reference(coords, radii, spec::JP.GridSpec{T};
                              w_rep, D, dpad) where {T}
    nx, ny, nz = spec.n
    R = zeros(T, nx, ny, nz)
    h = spec.spacing
    for n in 1:nz, m in 1:ny, l in 1:nx
        pt = spec.origin + SVector{3,T}((l - 1) * h, (m - 1) * h, (n - 1) * h)
        acc = zero(T)
        for a in eachindex(coords)
            r = norm(coords[a] - pt)
            d = radii[a] + T(dpad)
            if r < d
                acc += T(w_rep)
            elseif r < T(D)
                acc -= one(T)
            end
        end
        R[l, m, n] = acc
    end
    return R
end

@testset "shape scoring" begin
    T = Float64
    spec = JP.GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))

    @testset "receptor grid matches the direct definition" begin
        rng = Xoshiro(101)
        coords = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 3 for _ in 1:20]
        radii = T[1.4 + 0.5 * rand(rng) for _ in 1:20]
        R = Array{T,3}(undef, spec.n...)
        shape_receptor_grid!(R, coords, radii, spec; w_rep = 9, D = 6, dpad = 2)
        want = shape_grid_reference(coords, radii, spec; w_rep = 9, D = 6, dpad = 2)
        @test R == want
    end

    @testset "single atom: counts are the two concentric shells" begin

        c = spec.origin + SVector{3,T}(7.5 * spec.spacing, 7.5 * spec.spacing, 7.5 * spec.spacing)
        R = Array{T,3}(undef, spec.n...)
        shape_receptor_grid!(R, [c], T[1.7], spec; w_rep = 9, D = 6, dpad = 2)
        @test all(v -> v == 9 || v == -1 || v == 0, R)
        @test count(==(9), R) > 0
        @test count(==(-1), R) > 0

        h3 = spec.spacing^3
        @test count(==(9), R) * h3 ≈ (4 / 3) * pi * 3.7^3 rtol = 0.12
        @test (count(==(9), R) + count(==(-1), R)) * h3 ≈ (4 / 3) * pi * 6.0^3 rtol = 0.12

        @test maximum(R) > 0 && minimum(R) < 0
    end

    @testset "occupancy grid is the hard indicator" begin
        c = spec.origin + SVector{3,T}(7.5 * spec.spacing, 7.5 * spec.spacing, 7.5 * spec.spacing)
        L = Array{T,3}(undef, spec.n...)
        occupancy_grid!(L, [c], T[3.0], spec)
        @test all(v -> v == 0 || v == 1, L)
        @test count(==(1), L) * spec.spacing^3 ≈ (4 / 3) * pi * 27 rtol = 0.15

        L2 = similar(L)
        occupancy_grid!(L2, [c, c], T[3.0, 3.0], spec)
        @test L2 == L
    end

    @testset "correlate! with one shape term == direct correlation" begin

        n = 8
        sp = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-4.8, -4.8, -4.8))
        bk = JP.CPUBackend()
        rng = Xoshiro(202)
        rec = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 1.5 for _ in 1:10]
        lig = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 1.2 for _ in 1:6]
        radii = fill(T(1.7), length(rec))

        rs = shape_spectra(bk, rec, radii, sp; w_rep = 9, D = 6, flags = FFTW.ESTIMATE)
        @test JP.nterms(rs) == 1
        ws = JP.workspace(bk, sp, 1; batch = 1, flags = FFTW.ESTIMATE)
        th = 0.4
        rots = [SMatrix{3,3,T,9}(cos(th), sin(th), 0, -sin(th), cos(th), 0, 0, 0, 1)]
        got = copy(correlate!(ws, shape_structure(lig), rots, rs))

        Rg = Array{T,3}(undef, n, n, n)
        shape_receptor_grid!(Rg, rec, radii, sp; w_rep = 9, D = 6)
        Lg = Array{T}(undef, n, n, n, 1)
        JP.voxelize!(Lg, shape_structure(lig), rots, 1, sp)
        want = zeros(T, n, n, n)
        for c in 0:n-1, b in 0:n-1, a in 0:n-1
            s = zero(T)
            for k in 0:n-1, j in 0:n-1, i in 0:n-1
                s += Rg[i+1, j+1, k+1] *
                     Lg[mod(i+a, n)+1, mod(j+b, n)+1, mod(k+c, n)+1, 1]
            end
            want[a+1, b+1, c+1] = s
        end
        err = maximum(abs, got[:, :, :, 1] .- want) / maximum(abs, want)
        @test err < 1e-12
    end

    @testset "energy profile has the lock-and-key shape" begin

        big = JP.GridSpec{T}((32, 32, 32), T(1.2), SVector{3,T}(-19.2, -19.2, -19.2))
        rng = Xoshiro(303)
        rec = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 2.0 for _ in 1:60]
        radii = fill(T(1.7), length(rec))
        R = Array{T,3}(undef, big.n...)
        shape_receptor_grid!(R, rec, radii, big; w_rep = 9, D = 6)

        L = Array{T,3}(undef, big.n...)
        centre = SVector{3,T}(0, 0, 0)
        score(r) = begin
            occupancy_grid!(L, [centre + SVector{3,T}(r, 0, 0)], T[1.7], big)
            sum(R .* L)
        end
        rs_ = 0.0:1.0:16.0
        prof = [score(r) for r in rs_]

        @test prof[1] > 0
        @test minimum(prof) < 0
        @test prof[end] == 0
        @test argmax(prof) < argmin(prof)
        @test argmin(prof) < length(prof)

        rwell = rs_[argmin(prof)]
        @test 4 <= rwell <= 12
    end

    @testset "shape_structure and vdw_radius" begin
        coords = [SVector{3,T}(0, 0, 0), SVector{3,T}(1, 1, 1)]
        s = shape_structure(coords)
        @test JP.nterms(s) == 1
        @test all(s.termw .== 1)
        @test JP.natoms(s) == 2
        @test vdw_radius(Float64, "C") == 1.70
        @test vdw_radius(Float64, " n ") == 1.55
        @test vdw_radius(Float32, "XX") === 1.7f0
    end

    @testset "argument validation" begin
        R = Array{T,3}(undef, spec.n...)
        c = [SVector{3,T}(0, 0, 0)]
        @test_throws DimensionMismatch shape_receptor_grid!(R, c, T[1.0, 2.0], spec)
        bad = Array{T,3}(undef, 4, 4, 4)
        @test_throws DimensionMismatch shape_receptor_grid!(bad, c, T[1.7], spec)
        @test_throws DimensionMismatch occupancy_grid!(bad, c, T[1.7], spec)
        @test_throws DimensionMismatch spectra_from_grids(JP.CPUBackend(), [R], spec,
                                                          T[1.0, 2.0]; flags = FFTW.ESTIMATE)
    end
end

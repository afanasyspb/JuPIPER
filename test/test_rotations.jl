"""
Rotation-set generation and its quality metrics.

Two things are load-bearing here and neither is obvious from reading the code:

  * the **Hopf fibre property** -- every member of a fibre must point the
    ligand's own axis the same way, differing only by a spin about it. The
    textbook parametrisation gives the *other* coset and looks equally correct
    until you check. Any future direction-level pruning is built on this.
  * the **Float32 phase trap** -- the spiral phase reaches ~3e5 radians, where
    Float32 resolution is coarser than the grid being built. Generation must
    happen in Float64 whatever `T` is asked for.
"""

using Test, JuPIPER, StaticArrays, LinearAlgebra, Random
const JP = JuPIPER

@testset "rotation sets" begin

    @testset "generators produce valid SO(3) elements" begin
        for (name, rs) in (("fibonacci", rotations_fibonacci(Float64, 1024)),
                           ("hopf",      rotations_hopf(Float64, 64, 16)),
                           ("balanced",  rotations_hopf_balanced(Float64, 1024)),
                           ("random",    rotations_random(Float64, 512; rng = Xoshiro(7))))
            @test all(q -> abs(norm(q) - 1) < 1e-12, rs.quats)
            @test all(R -> maximum(abs, R' * R - I) < 1e-12, rs.rots)
            @test all(R -> abs(det(R) - 1) < 1e-12, rs.rots)
            @test length(rs) == length(rs.quats) == length(rs.rots)
        end
    end

    @testset "rotmatrix agrees with quaternion action" begin

        rng = Xoshiro(11)
        for _ in 1:50
            q = normalize(SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng)))
            v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
            w, x, y, z = q
            u = SVector{3,Float64}(x, y, z)

            uxv = cross(u, v)
            want = v + 2w * uxv + 2 * cross(u, uxv)
            @test JP.rotmatrix(q) * v ≈ want atol = 1e-13
        end
    end

    @testset "rotangle is a metric on SO(3), not on S^3" begin
        rng = Xoshiro(13)
        q = normalize(SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng)))
        p = normalize(SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng)))
        @test JP.rotangle(q, q) ≈ 0 atol = 1e-7
        @test JP.rotangle(q, -q) ≈ 0 atol = 1e-7
        @test JP.rotangle(q, p) ≈ JP.rotangle(p, q)
        @test JP.rotangle(q, p) ≈ JP.rotangle(-q, p)
        @test 0 <= JP.rotangle(q, p) <= pi + 1e-12

        Rrel = JP.rotmatrix(q)' * JP.rotmatrix(p)
        @test JP.rotangle(q, p) ≈ acos(clamp((tr(Rrel) - 1) / 2, -1, 1)) atol = 1e-9
    end

    @testset "Hopf fibre is a body-frame spin" begin
        ndir, nspin = 24, 12
        rs = rotations_hopf(Float64, ndir, nspin)
        @test rs.ndir == ndir && rs.nspin == nspin
        @test length(rs) == ndir * nspin
        xh = SVector(1.0, 0.0, 0.0)
        for i in (1, 7, ndir)
            f = fiber(rs, i)
            @test f == ((i - 1) * nspin + 1):(i * nspin)
            Rs = [rs.rots[k] for k in f]

            img = Rs[1] * xh
            @test all(R -> norm(R * xh - img) < 1e-12, Rs)

            for k in 1:(nspin - 1)
                Rrel = Rs[k]' * Rs[k+1]
                @test acos(clamp((tr(Rrel) - 1) / 2, -1, 1)) ≈ 2pi / nspin atol = 1e-10
                ax = SVector(Rrel[3,2] - Rrel[2,3], Rrel[1,3] - Rrel[3,1], Rrel[2,1] - Rrel[1,2])
                @test abs(abs(normalize(ax)[1]) - 1) < 1e-9
            end
        end

        dirs = [rs.rots[first(fiber(rs, i))] * xh for i in 1:ndir]
        @test minimum(norm(dirs[i] - dirs[j]) for i in 1:ndir for j in (i+1):ndir) > 1e-3
        @test_throws ArgumentError fiber(rotations_fibonacci(Float64, 16), 1)
        @test_throws BoundsError fiber(rs, ndir + 1)
    end

    @testset "sets are Haar-uniform" begin

        rng = Xoshiro(17)
        for rs in (rotations_fibonacci(Float64, 4096), rotations_hopf(Float64, 170, 24))
            for _ in 1:3
                q = normalize(SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng)))
                for w in (0.5, 1.0, 1.5, 2.0)
                    frac = count(p -> JP.rotangle(q, p) <= w, rs.quats) / length(rs)
                    @test frac ≈ (w - sin(w)) / pi atol = 0.03
                end
            end
        end
    end

    @testset "covering_bound inverts the ball measure" begin
        for n in (10, 100, 4096, 70_000)
            th = covering_bound(n)
            @test th - sin(th) ≈ pi / n rtol = 1e-10
            @test 0 < th < pi
        end

        @test covering_bound(1000) > covering_bound(10_000)
        @test covering_bound(10^6) ≈ cbrt(6pi / 10^6) rtol = 1e-3
        @test_throws ArgumentError covering_bound(0)
    end

    @testset "structured sets beat random sampling" begin

        n = 2048
        ns = 8_000
        fib = covering_radius(rotations_fibonacci(Float64, n); nsamples = ns, rng = Xoshiro(3))
        hop = covering_radius(rotations_hopf_balanced(Float64, n); nsamples = ns, rng = Xoshiro(3))
        rnd = covering_radius(rotations_random(Float64, n; rng = Xoshiro(5));
                              nsamples = ns, rng = Xoshiro(3))
        @test fib < rnd
        @test hop < rnd

        @test fib > covering_bound(n)
        @test hop > covering_bound(n)

        @test covering_efficiency(rotations_fibonacci(Float64, n);
                                  nsamples = ns, rng = Xoshiro(3)) > 0.5
    end

    @testset "hopf_split balances the two spacings" begin
        for n in (512, 4096, 70_000)
            ndir, nspin = hopf_split(n)
            @test ndir * nspin <= n
            @test ndir >= 1 && nspin >= 1

            @test 2pi / nspin ≈ sqrt(4pi / ndir) rtol = 0.15
        end
        @test hopf_split(70_000) == (1166, 60)
        @test length(rotations_hopf_balanced(Float64, 4096)) <= 4096
        @test_throws ArgumentError hopf_split(0)
    end

    @testset "Float32 sets are generated at Float64 precision" begin

        n = 70_000
        f32 = rotations_fibonacci(Float32, n)
        f64 = rotations_fibonacci(Float64, n)
        @test eltype(f32) === Float32
        @test maximum(maximum(abs, f32.quats[i] - Float32.(f64.quats[i])) for i in 1:n) <=
              4 * eps(Float32)

        h32 = rotations_hopf(Float32, 1166, 60)
        h64 = rotations_hopf(Float64, 1166, 60)
        @test maximum(maximum(abs, h32.quats[i] - Float32.(h64.quats[i]))
                      for i in eachindex(h32.quats)) <= 4 * eps(Float32)

        @test all(R -> maximum(abs, R' * R - I) < 1f-5, f32.rots[1:100])
    end

    @testset "nn_stats" begin
        rs = rotations_hopf(Float64, 128, 32)
        mn, md, mx = nn_stats(rs)
        @test 0 < mn <= md <= mx

        @test md ≈ 2pi / 32 atol = 1e-9
        @test_throws ArgumentError nn_stats(rotations_fibonacci(Float64, 1))
    end

    @testset "covering_spread brackets covering_radius" begin
        rs = rotations_hopf(Float64, 64, 16)
        lo, med, hi = covering_spread(rs; nsamples = 2_000, nrepeat = 5, seed = 2)
        @test lo <= med <= hi
        one_shot = covering_radius(rs; nsamples = 2_000, rng = Xoshiro(3))
        @test lo <= one_shot * 1.5
        @test hi <= pi
    end

    @testset "argument validation" begin
        @test_throws ArgumentError rotations_fibonacci(Float64, 0)
        @test_throws ArgumentError rotations_hopf(Float64, 0, 4)
        @test_throws ArgumentError rotations_hopf(Float64, 4, 0)
    end
end

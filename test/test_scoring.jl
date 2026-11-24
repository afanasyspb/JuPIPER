using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
using JuPIPER: GridSpec, Potential, CPUBackend, nterms, speclen, correlate!,
               workspace, spectra_from_grids

const PROFILES = load_cluspro_profiles()

@testset "ClusPro scoring profiles" begin

    @testset "Table 1 transcription and its six prose constraints" begin
        @test length(PROFILES) == 7
        @test [p.key for p in PROFILES] ==
              [:balanced, :electrostatic, :hydrophobic, :vdw_elec,
               :others1, :others2, :others3]

        bal  = cluspro_profile(:balanced;      profiles = PROFILES)
        elec = cluspro_profile(:electrostatic; profiles = PROFILES)
        hyd  = cluspro_profile(:hydrophobic;   profiles = PROFILES)
        vdw  = cluspro_profile(:vdw_elec;      profiles = PROFILES)
        o1   = cluspro_profile(:others1;       profiles = PROFILES)
        o2   = cluspro_profile(:others2;       profiles = PROFILES)
        o3   = cluspro_profile(:others3;       profiles = PROFILES)

        @test bal.w_elec == 600.0

        @test bal.w_dars == 1.0

        @test elec.w_elec == 2 * bal.w_elec
        @test (elec.w_rep, elec.w_attr, elec.w_dars) ==
              (bal.w_rep, bal.w_attr, bal.w_dars)

        @test hyd.w_dars == 2 * bal.w_dars
        @test (hyd.w_rep, hyd.w_attr, hyd.w_elec) ==
              (bal.w_rep, bal.w_attr, bal.w_elec)

        @test vdw.w_dars == 0.0
        @test (vdw.w_rep, vdw.w_attr, vdw.w_elec) ==
              (bal.w_rep, bal.w_attr, bal.w_elec)

        for o in (o1, o2, o3)
            @test abs(o.w_attr) < abs(bal.w_attr)
            @test o.w_elec < bal.w_elec
            @test o.w_elec == 300.0
        end
        @test length(unique((o1.w_dars, o2.w_dars, o3.w_dars))) == 3

        @test all(p -> p.w_attr < 0, PROFILES)
        @test all(p -> p.w_rep > 0, PROFILES)
    end

    @testset "loader rejects malformed rows" begin
        mktemp() do path, io
            write(io, "# header\nkey\tw_rep\tw_attr\tw_elec\tw_dars\tlabel\n" *
                      "bad\t0.4\t0.4\t600.0\t1.0\tsign lost\n")
            close(io)

            @test_throws ArgumentError load_cluspro_profiles(path)
        end
        mktemp() do path, io
            write(io, "key\tw_rep\tw_attr\tw_elec\tw_dars\tlabel\n" *
                      "dup\t0.4\t-0.4\t600.0\t1.0\tone\n" *
                      "dup\t0.4\t-0.4\t600.0\t1.0\ttwo\n")
            close(io)
            @test_throws ArgumentError load_cluspro_profiles(path)
        end
        @test_throws ArgumentError cluspro_profile(:no_such_mode; profiles = PROFILES)
    end

    @testset "shape ratio takes exactly two values" begin
        ratios = shape_ratio.(PROFILES)
        @test sort(unique(ratios)) ≈ [1.0, 2.5]
        @test count(≈(1.0), ratios) == 5
        @test count(≈(2.5), ratios) == 2

        @test Set(p.key for p in PROFILES if shape_ratio(p) ≈ 1.0) ==
              Set((:balanced, :electrostatic, :hydrophobic, :vdw_elec, :others1))
    end

    @testset "assumed_piper_weights: the arithmetic" begin
        bal = cluspro_profile(:balanced; profiles = PROFILES)
        w = assumed_piper_weights(bal)
        @test w isa PiperWeights{Float64}
        @test w.w1 ≈ bal.w_rep / abs(bal.w_attr)
        @test w.w2 ≈ bal.w_elec / abs(bal.w_attr)
        @test w.w3 ≈ bal.w_dars / abs(bal.w_attr)
        @test w.w1 ≈ 1.0 && w.w2 ≈ 1500.0 && w.w3 ≈ 2.5

        @test w.w1 < PIPER_WEIGHTS_ENZYME_INHIBITOR.w1 < 7.0

        w32 = assumed_piper_weights(ClusProProfile{Float32}(bal))
        @test w32 isa PiperWeights{Float32}
        @test w32.w1 ≈ Float32(w.w1)

        degenerate = ClusProProfile{Float64}(:x, "x", 0.4, 0.0, 600.0, 1.0)
        @test_throws ArgumentError assumed_piper_weights(degenerate)
    end

    @testset "the conversion is rank-preserving, checked on a landscape" begin

        rng = MersenneTwister(9041)
        T = Float64
        n = (16, 16, 16)
        ncell = prod(n)

        E_rep  = rand(rng, T, ncell)
        E_attr = rand(rng, T, ncell)
        E_elec = randn(rng, T, ncell)
        E_dars = randn(rng, T, ncell)

        for p in PROFILES
            E2017 = @. p.w_rep * E_rep + p.w_attr * E_attr +
                       p.w_elec * E_elec + p.w_dars * E_dars
            w = assumed_piper_weights(p)

            Eours = @. (w.w1 * E_rep - E_attr) + w.w2 * E_elec + w.w3 * E_dars
            @test Eours ≈ E2017 ./ abs(p.w_attr)
            @test sortperm(Eours) == sortperm(E2017)
            @test argmin(Eours) == argmin(E2017)
        end
    end

    @testset "retarget_spectra! equals a rebuild from scratch" begin
        rng = MersenneTwister(4471)
        T = Float64
        P = 3
        spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(0, 0, 0))
        pot = load_dars()
        potT = Potential{T}(pot)

        grids = [randn(rng, T, spec.n...) for _ in 1:(2+P)]
        wfrom = PiperWeights{T}(2.1, 133.0, 2.2)
        wto   = PiperWeights{T}(2.1, 600.0, 0.75)

        mk(w) = spectra_from_grids(CPUBackend(), grids, spec,
                                   receptor_weights(potT, w; P = P);
                                   flags = FFTW.ESTIMATE)
        rs   = mk(wfrom)
        want = mk(wto)

        buf = Vector{T}(undef, 2 + P)
        retarget_spectra!(rs, buf, potT, wfrom, wto; P = P)
        for p in 1:(2+P)

            @test rs.data[p] ≈ want.data[p] rtol = 1e-13
        end

        retarget_spectra!(rs, buf, potT, wto, wfrom; P = P)
        back = mk(wfrom)
        for p in 1:(2+P)
            @test rs.data[p] ≈ back.data[p] rtol = 1e-13
        end
    end

    @testset "term_ratio! refuses what it cannot express" begin
        T = Float64
        P = 3
        potT = Potential{T}(load_dars())
        buf = Vector{T}(undef, 2 + P)

        w0 = PiperWeights{T}(2.1, 133.0, 2.2)

        @test_throws ArgumentError term_ratio!(buf, potT, w0,
                                               PiperWeights{T}(2.5, 133.0, 2.2); P = P)

        wz = PiperWeights{T}(2.1, 133.0, 0.0)
        @test_throws ArgumentError term_ratio!(buf, potT, wz, w0; P = P)

        @test_throws DimensionMismatch term_ratio!(Vector{T}(undef, 2), potT, w0, w0; P = P)

        term_ratio!(buf, potT, w0, PiperWeights{T}(2.1, 133.0, 0.0); P = P)
        @test buf[1] == 1
        @test buf[2] == 1
        @test all(iszero, @view buf[3:end])

        term_ratio!(buf, potT, w0, PiperWeights{T}(2.1, 266.0, 4.4); P = P)
        @test buf[2] ≈ 2
        @test all(≈(2), @view buf[3:end])
    end

    @testset "switching a profile allocates nothing" begin

        T = Float64
        P = 3
        spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(0, 0, 0))
        potT = Potential{T}(load_dars())
        grids = [randn(T, spec.n...) for _ in 1:(2+P)]
        wa = PiperWeights{T}(2.1, 133.0, 2.2)
        wb = PiperWeights{T}(2.1, 600.0, 0.75)
        rs = spectra_from_grids(CPUBackend(), grids, spec,
                                receptor_weights(potT, wa; P = P);
                                flags = FFTW.ESTIMATE)
        buf = Vector{T}(undef, 2 + P)

        function switch_bytes(rs, buf, potT, wa, wb, P)
            retarget_spectra!(rs, buf, potT, wa, wb; P = P)
            retarget_spectra!(rs, buf, potT, wb, wa; P = P)
            return @allocated retarget_spectra!(rs, buf, potT, wa, wb; P = P)
        end
        @test switch_bytes(rs, buf, potT, wa, wb, P) == 0

        function rescale_bytes(rs, f)
            rescale_terms!(rs, f)
            return @allocated rescale_terms!(rs, f)
        end
        @test rescale_bytes(rs, fill(one(T), 2 + P)) == 0
        @test rescale_bytes(rs, fill(T(1.5), 2 + P)) == 0
        @test_throws DimensionMismatch rescale_terms!(rs, ones(T, 2))
    end

    @testset "a re-weighted run equals a rebuilt run, end to end" begin

        rng = MersenneTwister(20250812)
        T = Float64
        P = 2
        n = (16, 16, 16)
        spec = GridSpec{T}(n, T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
        potT = Potential{T}(load_dars())
        grids = [randn(rng, T, n...) for _ in 1:(2+P)]
        coords = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 3 for _ in 1:24]
        lig = JuPIPER.Structure(coords, randn(rng, T, 24, 2 + P))
        rots = [SMatrix{3,3,T,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)]

        wa = PiperWeights{T}(2.1, 133.0, 2.2)
        wb = PiperWeights{T}(2.1, 1500.0, 2.5)

        mk(w) = spectra_from_grids(CPUBackend(), grids, spec,
                                   receptor_weights(potT, w; P = P);
                                   flags = FFTW.ESTIMATE)
        ws = workspace(CPUBackend(), spec, 2 + P; batch = 1, flags = FFTW.ESTIMATE)

        rs_ref = mk(wb)
        correlate!(ws, lig, rots, rs_ref)
        want = copy(ws.score)

        rs_sw = mk(wa)
        buf = Vector{T}(undef, 2 + P)
        retarget_spectra!(rs_sw, buf, potT, wa, wb; P = P)
        correlate!(ws, lig, rots, rs_sw)
        @test ws.score ≈ want rtol = 1e-12

        @test argmin(want) == argmin(ws.score)
    end
end

using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random, Printf
using JuPIPER: GridSpec, Structure, Potential, natoms, speclen, build_termweights,
               voxelize!, CPUBackend, nterms
using BioStructures


gridpoint(spec::GridSpec{T}, l, m, n) where {T} =
    spec.origin + SVector{3,T}((l - 1) * spec.spacing, (m - 1) * spec.spacing,
                               (n - 1) * spec.spacing)

"""Generalized Born receptor field, PDF p.11, both readings of the cutoff."""
function gb_reference(coords, q, spec::GridSpec{T}; D, capped::Bool) where {T}
    nx, ny, nz = spec.n
    R = zeros(T, nx, ny, nz)
    rcut = 2 * T(D)
    D2 = T(D)^2
    den(r) = sqrt(r^2 + D2 * exp(-r^2 / (4 * D2)))
    for n in 1:nz, m in 1:ny, l in 1:nx
        pt = gridpoint(spec, l, m, n)
        acc = zero(T)
        for a in eachindex(coords)
            r = norm(coords[a] - pt)
            if capped
                acc += q[a] / den(min(r, rcut))
            elseif r < rcut
                acc += q[a] / den(r)
            end
        end
        R[l, m, n] = acc
    end
    return R
end

"""Pairwise delta field, PDF p.13: `R_p = sum_i u_pI * [r_i < D]`."""
function dars_reference(coords, u, spec::GridSpec{T}; D) where {T}
    nx, ny, nz = spec.n
    R = zeros(T, nx, ny, nz)
    for n in 1:nz, m in 1:ny, l in 1:nx
        pt = gridpoint(spec, l, m, n)
        acc = zero(T)
        for a in eachindex(coords)
            norm(coords[a] - pt) < T(D) && (acc += u[a])
        end
        R[l, m, n] = acc
    end
    return R
end

"""True 3-D separable convolution with zero padding -- what `box_smooth!` claims."""
function box_reference(A::Array{T,3}, offs, wts) where {T}
    nx, ny, nz = size(A)
    B = zeros(T, nx, ny, nz)
    for n in 1:nz, m in 1:ny, l in 1:nx
        acc = zero(T)
        for (tk, dk) in enumerate(offs), (tj, dj) in enumerate(offs),
            (ti, di) in enumerate(offs)
            i = l + di; j = m + dj; k = n + dk
            (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
            acc += wts[ti] * wts[tj] * wts[tk] * A[i, j, k]
        end
        B[l, m, n] = acc
    end
    return B
end

"""Direct periodic correlation of explicit grids, the `test_correctness.jl` oracle."""
function direct_energy(grids, weights, ligrids, spec::GridSpec{T}, a, b, c) where {T}
    nx, ny, nz = spec.n
    tot = zero(T)
    for p in eachindex(grids)
        s = zero(T)
        for k in 1:nz, j in 1:ny, i in 1:nx
            ii = mod1(i + a, nx); jj = mod1(j + b, ny); kk = mod1(k + c, nz)
            s += grids[p][i, j, k] * ligrids[p][ii, jj, kk]
        end
        tot += weights[p] * s
    end
    return tot
end


function pdbline(serial, name, resname, chain, resnum, x, y, z, elem)
    nm = length(strip(elem)) == 2 ? rpad(name, 4) : rpad(" " * name, 4)
    return string("ATOM  ", lpad(serial, 5), " ", nm, " ", rpad(resname, 3), " ",
                  chain, lpad(resnum, 4), "    ",
                  @sprintf("%8.3f%8.3f%8.3f", x, y, z),
                  @sprintf("%6.2f%6.2f", 1.0, 20.0), " "^10, lpad(elem, 2))
end

function write_small_fixture(path)
    open(path, "w") do io
        s = 0
        nxt() = (s += 1)

        for (nm, x, y, z, e) in (("N", 0.0, 0.0, 0.0, "N"), ("CA", 1.46, 0.0, 0.0, "C"),
                                 ("C", 2.0, 1.42, 0.0, "C"), ("O", 1.3, 2.4, 0.2, "O"))
            println(io, pdbline(nxt(), nm, "GLY", "A", 1, x, y, z, e))
        end

        for (nm, x, y, z, e) in (("N", 3.3, 1.5, 0.0, "N"), ("CA", 4.0, 2.8, 0.0, "C"),
                                 ("C", 5.5, 2.6, 0.1, "C"), ("O", 6.1, 1.5, 0.1, "O"),
                                 ("CB", 3.6, 3.7, 1.2, "C"), ("CG", 4.1, 5.1, 1.0, "C"),
                                 ("OD1", 3.7, 5.9, 1.9, "O"), ("OD2", 4.8, 5.4, 0.0, "O"))
            println(io, pdbline(nxt(), nm, "ASP", "A", 2, x, y, z, e))
        end

        for (nm, x, y, z, e) in (("N", 6.1, 3.7, 0.2, "N"), ("CA", 7.5, 3.7, 0.3, "C"),
                                 ("C", 8.0, 5.1, 0.4, "C"), ("O", 7.3, 6.1, 0.4, "O"),
                                 ("CB", 8.1, 3.0, 1.5, "C"), ("OXT", 9.2, 5.2, 0.5, "O"))
            println(io, pdbline(nxt(), nm, "ALA", "A", 3, x, y, z, e))
        end
        println(io, "END")
    end
    return path
end


@testset "physics: voxelisation and receptor fields" begin
    T = Float64
    spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    rng = Xoshiro(2026)
    coords = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 2.5 for _ in 1:24]
    charges = T[randn(rng) for _ in 1:24]
    charges .-= sum(charges) / length(charges)
    uvec = T[randn(rng) for _ in 1:24]

    @testset "box smoothing" begin
        offs, wts = box_taps(T, PIPER_SMOOTH_BOX, PIPER_SPACING)

        @test offs == [-1, 0, 1]
        @test wts ≈ T[0.3, 0.4, 0.3]
        @test sum(wts) ≈ one(T)
        @test wts[1] == wts[3]

        for L in (T(1.2), T(0.9), T(0.1))
            o, w = box_taps(T, L, T(1.2))
            @test o == [0] && w ≈ T[1]
        end

        for L in (T(2.4), T(3.6), T(6.0), T(7.1))
            o, w = box_taps(T, L, T(1.2))
            @test sum(w) ≈ one(T)
            @test w ≈ reverse(w)
            @test o == -maximum(o):maximum(o)
        end
        @test_throws ArgumentError box_taps(T, 0.0, 1.2)

        A = rand(rng, T, spec.n...)
        want = box_reference(copy(A), offs, wts)
        got = box_smooth!(copy(A), spec)
        @test got ≈ want rtol = 64 * eps(T)

        @test box_smooth!(copy(A), box_filter(T, spec)) == got

        C = fill(T(3), spec.n...)
        box_smooth!(C, spec)
        @test all(≈(T(3)), C[2:end-1, 2:end-1, 2:end-1])
        @test C[1, 8, 8] < T(3)

        B0 = copy(A)
        @test box_smooth!(copy(B0), spec; edge = T(1.2)) == B0

        Rs = Array{T,3}(undef, spec.n...); Rn = similar(Rs)
        gb_receptor_grid!(Rs, coords, charges, spec)
        gb_receptor_grid!(Rn, coords, charges, spec; smooth = nothing)
        @test Rs != Rn
        @test box_smooth!(copy(Rn), spec) ≈ Rs

        f = box_filter(T, spec)
        @test f.edge == T(PIPER_SMOOTH_BOX)
        @test length(f.line) == maximum(spec.n)
        @test occursin("BoxFilter", sprint(show, f))
    end

    @testset "Generalized Born field vs PDF p.11" begin
        R = Array{T,3}(undef, spec.n...)

        atol = 64 * eps(T) * sum(abs, charges) / T(PIPER_D)

        gb_receptor_grid!(R, coords, charges, spec; smooth = nothing, truncation = GBCapped())
        want_cap = gb_reference(coords, charges, spec; D = PIPER_D, capped = true)
        @test maximum(abs, R .- want_cap) <= atol

        gb_receptor_grid!(R, coords, charges, spec; smooth = nothing, truncation = GBTruncated())
        want_tru = gb_reference(coords, charges, spec; D = PIPER_D, capped = false)
        @test maximum(abs, R .- want_tru) <= atol

        d = want_cap .- want_tru
        @test maximum(d) - minimum(d) > 1e-3

        rcut = 2 * T(PIPER_D)
        allnear = [all(a -> norm(a - gridpoint(spec, l, m, n)) < rcut, coords)
                   for l in 1:16, m in 1:16, n in 1:16]
        @test any(allnear)
        off = sum(charges) / sqrt(rcut^2 + T(PIPER_D)^2 * exp(-rcut^2 / (4 * T(PIPER_D)^2)))
        @test all(abs.(d[allnear] .- off) .<= atol)

        one_at = [SVector{3,T}(0, 0, 0)]
        fine = GridSpec{T}((64, 4, 4), T(0.4), SVector{3,T}(-0.6, -0.6, -0.6))
        Rc = Array{T,3}(undef, fine.n...); Rt = similar(Rc)
        gb_receptor_grid!(Rc, one_at, T[1], fine; smooth = nothing, truncation = GBCapped())
        gb_receptor_grid!(Rt, one_at, T[1], fine; smooth = nothing, truncation = GBTruncated())
        line_c = [Rc[l, 2, 2] for l in 1:64]
        line_t = [Rt[l, 2, 2] for l in 1:64]
        @test maximum(abs, diff(line_c)) < 5e-3
        @test maximum(abs, diff(line_t)) > 5e-2

        zc = zeros(T, length(coords))
        gb_receptor_grid!(R, coords, zc, spec; smooth = nothing)
        @test all(iszero, R)
        gb_receptor_grid!(R, coords, zc, spec)
        @test all(iszero, R)
    end

    @testset "DARS delta field vs PDF p.13" begin
        R = Array{T,3}(undef, spec.n...)
        dars_receptor_grid!(R, coords, uvec, spec)
        want = dars_reference(coords, uvec, spec; D = PIPER_D)
        @test maximum(abs, R .- want) <= 64 * eps(T) * sum(abs, uvec)

        ones_u = ones(T, length(coords))
        dars_receptor_grid!(R, coords, ones_u, spec)
        @test all(x -> x == round(x), R)
        centre = gridpoint(spec, 8, 8, 8)
        @test R[8, 8, 8] == count(a -> norm(a - centre) < T(PIPER_D), coords)

        @test_throws DimensionMismatch dars_receptor_grid!(R, coords, uvec[1:3], spec)
        @test_throws DimensionMismatch dars_receptor_grid!(
            Array{T,3}(undef, 8, 8, 8), coords, uvec, spec)
    end

    @testset "mass conservation" begin
        nt = 3
        W = Matrix{T}(undef, length(coords), nt)
        W[:, 1] .= one(T)
        W[:, 2] .= charges
        W[:, 3] .= uvec
        s = Structure(copy(coords), W)
        dst = Array{T,4}(undef, spec.n..., 1)

        fit = grid_fit(spec, s.coords)
        @test fit.n_leaking == 0
        @test fit.leak_total == 0
        @test fit.margin_cells > 0
        @test fit.margin_angstrom ≈ fit.margin_cells * spec.spacing

        for t in 1:nt
            voxelize!(dst, s, [one(SMatrix{3,3,T,9})], t, spec)
            md = mass_defect(dst, s, t)
            @test md.ok
            @test md.reldefect[1] <= md.tol

            @test md.tol < 1e-13
        end

        shifted = [c + SVector{3,T}(8.6, 0, 0) for c in coords]
        s2 = Structure(shifted, W)
        f2 = grid_fit(spec, shifted)
        @test f2.n_leaking > 0
        @test f2.leak_worst > 0
        @test 1 <= f2.worst_atom <= length(coords)
        for t in 1:nt
            voxelize!(dst, s2, [one(SMatrix{3,3,T,9})], t, spec)
            predicted = sum(W[a, t] * (1 - first(stencil_leak(spec, shifted[a])))
                            for a in eachindex(shifted))
            @test sum(dst) ≈ predicted atol = mass_tolerance(T, length(coords),
                                                             prod(spec.n)) *
                                              sum(abs, @view W[:, t])
        end

        far = [SVector{3,T}(1000, 0, 0)]
        @test first(stencil_leak(spec, far[1])) == 1
        ffit = grid_fit(spec, far)
        @test ffit.n_outside == 1 && ffit.leak_total == 1 && ffit.margin_cells < 0

        for _ in 1:50
            p = SVector{3,T}(rand(rng) * 6 - 3, rand(rng) * 6 - 3, rand(rng) * 6 - 3)
            @test first(stencil_leak(spec, p)) ≈ 0 atol = 8 * eps(T)
        end

        big = GridSpec{T}((64, 64, 64), T(1.2), SVector{3,T}(-38.4, -38.4, -38.4))
        rfit = grid_fit_rotated(big, s.coords)
        @test rfit.margin_cells > 1 && rfit.n_leaking == 0
        tight = GridSpec{T}((8, 8, 8), T(1.2), SVector{3,T}(-4.8, -4.8, -4.8))
        @test grid_fit_rotated(tight, s.coords).n_leaking == length(coords)

        rots = rotations_fibonacci(T, 200)
        for R in rots.rots
            @test grid_fit(big, [R * c for c in s.coords]).n_leaking == 0
        end

        @test grid_fit(spec, SVector{3,T}[]).natoms == 0
        @test grid_fit_rotated(spec, SVector{3,T}[]).natoms == 0
    end

    @testset "choosing P: truncation error and dars_rank" begin
        pot = load_dars(); tblacp = load_acp_types()

        K = length(pot.labels)
        uni = fill(1 / K, K)
        for P in 1:6
            @test truncation_error(pot, P)[2] ≈ truncation_error(pot, P; weights = uni)[2]
        end

        errs = [truncation_error(pot, P)[2] for P in 1:K]
        @test issorted(errs; rev = true)
        @test errs[end] < 1e-12
        @test_throws DimensionMismatch truncation_error(pot, 2; weights = [1.0, 2.0])

        ab = type_abundance(tblacp)
        @test length(ab) == K
        @test sum(ab) ≈ 1
        @test all(>(0), ab)

        @test sum(ab) * JuPIPER.ACP_STANDARD_ATOM_COUNT ≈ 167

        for tol in (0.30, 0.20, 0.10, 0.05)
            P = dars_rank(pot; tol = tol)
            @test truncation_error(pot, P)[2] <= tol
            P > 1 && @test truncation_error(pot, P - 1)[2] > tol
        end

        @test truncation_error(pot, 4)[2] > 0.10
        @test dars_rank(pot) == 5
        @test dars_rank(pot; weights = ab) == 5
        @test dars_rank(pot; tol = 1e-9, Pmax = 6) == 6
    end

    @testset "receptor weights and term order" begin
        pot = load_dars()
        w = PIPER_WEIGHTS_ENZYME_INHIBITOR
        for P in 1:4
            v = receptor_weights(pot, w; P = P)
            @test length(v) == 2 + P
            @test v[1] == 1
            @test v[2] == w.w2
            @test v[3:end] ≈ w.w3 .* pot.lambda[1:P]
        end
        @test_throws ArgumentError receptor_weights(pot, w; P = 0)
        @test_throws ArgumentError receptor_weights(pot, w; P = 19)
    end

    @testset "assembled spectra reproduce the direct correlation" begin

        small = GridSpec{T}((8, 8, 8), T(1.2), SVector{3,T}(-4.8, -4.8, -4.8))
        rc = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 1.5 for _ in 1:10]
        lc = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) .* 1.2 for _ in 1:7]
        qr = T[randn(rng) for _ in 1:10]
        ql = T[randn(rng) for _ in 1:7]
        radii = T[1.7 for _ in 1:10]
        pot = load_dars()
        P = 2
        types = rand(rng, 1:18, 7)

        grids = [Array{T,3}(undef, small.n...) for _ in 1:(2+P)]
        shape_receptor_grid!(grids[1], rc, radii, small)
        gb_receptor_grid!(grids[2], rc, qr, small)
        for p in 1:P
            u = T[pot.U[rand(rng, 1:18), p] for _ in 1:10]
            dars_receptor_grid!(grids[2+p], rc, u, small)
        end
        wts = receptor_weights(pot, PIPER_WEIGHTS_ENZYME_INHIBITOR; P = P)
        rs = spectra_from_grids(CPUBackend(), grids, small, wts;
                                flags = FFTW.ESTIMATE)

        W = build_termweights(types, ql, view(pot.U, :, 1:P))
        lig = Structure(lc, W)
        ws = workspace(CPUBackend(), small, 2 + P; batch = 1, flags = FFTW.ESTIMATE)
        correlate!(ws, lig, [one(SMatrix{3,3,T,9})], rs)

        ligrids = [Array{T,3}(undef, small.n...) for _ in 1:(2+P)]
        tmp = Array{T,4}(undef, small.n..., 1)
        for p in 1:(2+P)
            voxelize!(tmp, lig, [one(SMatrix{3,3,T,9})], p, small)
            ligrids[p] .= @view tmp[:, :, :, 1]
        end
        scale = maximum(p -> sum(abs, grids[p]) * abs(wts[p]) * sum(abs, ligrids[p]),
                        1:(2+P))
        for (a, b, c) in ((0, 0, 0), (1, 0, 0), (3, 5, 2), (7, 7, 7), (4, 2, 6))
            want = direct_energy(grids, wts, ligrids, small, a, b, c)
            @test ws.score[a+1, b+1, c+1, 1] ≈ want atol = 1e-9 * max(scale, one(T))
        end
    end

    @testset "grid builders allocate nothing" begin

        R = Array{T,3}(undef, spec.n...)
        filt = box_filter(T, spec)
        alloc_gb(R, c, q, s, f) = (gb_receptor_grid!(R, c, q, s; smooth = f);
                                   @allocated gb_receptor_grid!(R, c, q, s; smooth = f))
        alloc_gbt(R, c, q, s, f) =
            (gb_receptor_grid!(R, c, q, s; truncation = GBTruncated(), smooth = f);
             @allocated gb_receptor_grid!(R, c, q, s; truncation = GBTruncated(), smooth = f))
        alloc_dars(R, c, u, s) = (dars_receptor_grid!(R, c, u, s);
                                  @allocated dars_receptor_grid!(R, c, u, s))
        alloc_smooth(R, f) = (box_smooth!(R, f); @allocated box_smooth!(R, f))
        alloc_fit(s, c) = (grid_fit(s, c); @allocated grid_fit(s, c))
        alloc_rfit(s, c) = (grid_fit_rotated(s, c); @allocated grid_fit_rotated(s, c))
        alloc_leak(s, p) = (stencil_leak(s, p); @allocated stencil_leak(s, p))

        @test alloc_gb(R, coords, charges, spec, filt) == 0
        @test alloc_gbt(R, coords, charges, spec, filt) == 0
        @test alloc_dars(R, coords, uvec, spec) == 0
        @test alloc_smooth(R, filt) == 0
        @test alloc_fit(spec, coords) == 0
        @test alloc_rfit(spec, coords) == 0
        @test alloc_leak(spec, coords[1]) == 0
    end

    @testset "Float32 path agrees with Float64" begin
        S = Float32
        spec32 = GridSpec{S}(spec.n, S(spec.spacing), SVector{3,S}(spec.origin))
        c32 = [SVector{3,S}(c) for c in coords]
        q32 = S.(charges); u32 = S.(uvec)
        R32 = Array{S,3}(undef, spec.n...); R64 = Array{T,3}(undef, spec.n...)

        gb_receptor_grid!(R32, c32, q32, spec32)
        gb_receptor_grid!(R64, coords, charges, spec)
        @test eltype(R32) === Float32
        @test maximum(abs, R32 .- S.(R64)) <= 64 * eps(S) * sum(abs, q32) / S(PIPER_D)

        dars_receptor_grid!(R32, c32, u32, spec32)
        dars_receptor_grid!(R64, coords, uvec, spec)
        @test maximum(abs, R32 .- S.(R64)) <= 64 * eps(S) * sum(abs, u32)

        @test mass_tolerance(Float32, 2000, 128^3) > mass_tolerance(Float64, 2000, 128^3)
        @test box_taps(Float32, 3.0, 1.2)[2] ≈ Float32[0.3, 0.4, 0.3]
    end

    @testset "end to end from a PDB file (2.1 -> 2.4)" begin
        dir = mktempdir()
        path = write_small_fixture(joinpath(dir, "tri.pdb"))
        at = read_atoms(path; T = T)
        @test length(at) == 18
        @test at.centred

        tblacp = load_acp_types()
        pot = load_dars()
        par = parametrize(tblacp, at; charges = FormalCharges())
        @test isempty(par.untyped)
        @test isempty(par.unknown_element)

        at2 = read_atoms(path; T = T)
        gs = gridspec_for(at, at2)
        @test gs == gridspec_for(at.coords, at2.coords)
        @test gs.spacing == T(PIPER_SPACING)
        ext_r = extent(at); ligmax = maximum(extent(at2))
        for ax in 1:3
            @test gs.n[ax] == nextfftsize(ceil(Int, (ext_r[ax] + ligmax) / 1.2 + 2))
        end
        @test grid_fit(gs, at).n_leaking == 0

        gspec = GridSpec{T}((32, 32, 32), T(1.2), SVector{3,T}(-19.2, -19.2, -19.2))
        fit = grid_fit(gspec, at)
        @test fit.n_leaking == 0
        @test fit.margin_angstrom > PIPER_D

        P = 2
        grids = [Array{T,3}(undef, gspec.n...) for _ in 1:(2+P)]
        piper_receptor_grids!(grids, at, par, pot, gspec; P = P)

        want = Array{T,3}(undef, gspec.n...)
        shape_receptor_grid!(want, at.coords, par.radius, gspec)
        @test grids[1] == want
        gb_receptor_grid!(want, at.coords, par.charge, gspec)
        @test grids[2] == want
        for p in 1:P
            u = T[pot.U[par.acptype[a], p] for a in 1:length(at)]
            dars_receptor_grid!(want, at.coords, u, gspec)
            @test grids[2+p] == want
        end

        @test !all(iszero, grids[2])
        @test any(<(0), grids[2])

        rs = @test_logs (:warn,) piper_receptor_spectra(CPUBackend(), at, par, pot,
                                                        gspec; P = P,
                                                        flags = FFTW.ESTIMATE)
        @test nterms(rs) == 2 + P
        @test rs.spec == gspec

        @test_logs (:warn,) piper_receptor_spectra(CPUBackend(), at, par, pot,
                                                   gspec; P = 1, flags = FFTW.ESTIMATE)

        @test_logs piper_receptor_spectra(CPUBackend(), at, par, pot, gspec;
                                          P = 5, flags = FFTW.ESTIMATE)

        lig = piper_structure(at, par, pot)
        @test nterms(lig) == 2 + dars_rank(pot)
        @test nterms(lig) == 7

        @test_throws DimensionMismatch piper_receptor_grids!(grids, at, par, pot,
                                                             gspec; P = P + 1)
        rm(dir; recursive = true)
    end
end

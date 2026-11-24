using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random, Printf
using JuPIPER: GridSpec, Structure, Pose, CPUBackend, receptor_spectra, second_moment
using BioStructures


function vline(serial, name, resname, chain, resnum, x, y, z, elem)
    nm = length(strip(elem)) == 2 ? rpad(name, 4) : rpad(" " * name, 4)
    return string("ATOM  ", lpad(serial, 5), " ", nm, " ", rpad(resname, 3), " ",
                  chain, lpad(resnum, 4), " ", "   ",
                  @sprintf("%8.3f%8.3f%8.3f", x, y, z),
                  @sprintf("%6.2f%6.2f", 1.0, 0.0), " "^10, lpad(elem, 2))
end

"""
Two chains of ALA residues, each residue a rigid 5-atom cluster centred where we
put it. Chain A at x = 0, 20, 40; chain B at x = 5, 60. At a 10 A cutoff exactly
one residue of each is at the interface -- A1 to B1 -- and every other pair is
15 A or further. That makes `interface_masks` checkable by counting rather than
by trusting.
"""
function write_two_chain(path)
    open(path, "w") do io
        serial = 0
        for (chain, centres) in (("A", (0.0, 20.0, 40.0)), ("B", (5.0, 60.0)))
            for (ri, cx) in enumerate(centres)
                for (nm, el, dx, dy, dz) in (("N", "N", -1.2, 0.0, 0.0),
                                             ("CA", "C", 0.0, 0.0, 0.0),
                                             ("C", "C", 1.2, 0.0, 0.0),
                                             ("O", "O", 1.2, 1.1, 0.0),
                                             ("CB", "C", 0.0, -1.0, 1.0))
                    serial += 1
                    println(io, vline(serial, nm, "ALA", chain, ri,
                                      cx + dx, dy, dz, el))
                end
            end
        end
        println(io, "END")
    end
    return path
end

const VDIR = mktempdir()
const VFIX = write_two_chain(joinpath(VDIR, "twochain.pdb"))

@testset "validation" begin


@testset "minimal image" begin
    for n in (8, 16, 66, 67)
        for t in 0:(n-1)
            m = minimal_image(t, n)
            @test mod(m, n) == t
            @test -n / 2 < m <= n / 2
        end
    end
    @test minimal_image(0, 8) == 0
    @test minimal_image(4, 8) == 4
    @test minimal_image(5, 8) == -3

    @test minimal_image(20, 8) == 4
    @test minimal_image(-1, 8) == -1
end


@testset "pose transform convention" begin

    T = Float64
    spec = GridSpec{T}((16, 16, 16), T(1.0), SVector{3,T}(-8, -8, -8))
    for p_r in (SVector{3,T}(2, -3, 1), SVector{3,T}(-5, 0, 6), SVector{3,T}(0, 0, 0))
        rec = Structure([p_r], reshape(T[1], 1, 1))
        lig = Structure([zero(SVector{3,T})], reshape(T[1], 1, 1))
        rs = receptor_spectra(CPUBackend(), rec, spec, T[-1]; flags = FFTW.ESTIMATE)
        ws = workspace(CPUBackend(), spec, 1; batch = 1, flags = FFTW.ESTIMATE)
        rots = [one(SMatrix{3,3,T,9})]
        poses = Vector{Pose{T}}(undef, 1)
        dock!(poses, lig, rots, rs, [ws])
        R, t = pose_transform(poses[1], rots, spec)
        @test R == one(SMatrix{3,3,T,9})
        @test pose_coords(lig.coords, R, t)[1] ≈ p_r atol = 1e-12
    end
end


@testset "matched-filter recovery, end to end" begin

    T = Float64
    rng = MersenneTwister(2_5_2026)
    n = 24
    spec = GridSpec{T}((n, n, n), T(1.0), SVector{3,T}(-12, -12, -12))
    natom = 40
    y = [SVector{3,T}(4randn(rng), 4randn(rng), 4randn(rng)) for _ in 1:natom]
    y .-= (sum(y) / natom,)
    W = ones(T, natom, 1)
    lig = Structure(y, W)

    rset = rotations_hopf(T, 4, 4)
    r0 = 7
    R0 = rset.rots[r0]
    cells = (2, -3, 1)
    t0 = -spec.spacing * SVector{3,T}(cells...)
    placed = [R0 * yk + t0 for yk in y]

    rec = Structure(placed, ones(T, natom, 1))
    rs = receptor_spectra(CPUBackend(), rec, spec, T[-1]; flags = FFTW.ESTIMATE)
    ws = workspace(CPUBackend(), spec, 1; batch = 1, flags = FFTW.ESTIMATE)
    poses = Vector{Pose{T}}(undef, length(rset))
    dock!(poses, lig, rset.rots, rs, [ws])

    order = sortperm(poses; by = p -> p.energy)
    best = poses[order[1]]
    @test Int(best.rot) == r0
    R, t = pose_transform(best, rset.rots, spec)
    @test R === R0
    @test t ≈ t0 atol = 1e-12
    @test ligand_rmsd(pose_coords(y, R, t), placed) < 1e-12

    fl = search_floor(rset, spec, y, placed)
    @test fl.rmsd < 1e-12
    @test fl.rot == r0
    @test fl.cells == cells
end


@testset "search floor vs brute force" begin
    T = Float64
    rng = MersenneTwister(99)
    spec = GridSpec{T}((32, 32, 32), T(1.2), SVector{3,T}(-19.2, -19.2, -19.2))
    y = [SVector{3,T}(3randn(rng), 3randn(rng), 3randn(rng)) for _ in 1:9]
    y .-= (sum(y) / 9,)
    z = [SVector{3,T}(2.3, -4.1, 0.7) + SVector{3,T}(3randn(rng), 3randn(rng), 3randn(rng))
         for _ in 1:9]
    rset = rotations_hopf(T, 5, 3)

    fl = search_floor(rset, spec, y, z)

    best = Inf
    for r in eachindex(rset.rots), a in -8:8, b in -8:8, c in -8:8
        R = rset.rots[r]
        t = -spec.spacing * SVector{3,T}(a, b, c)
        s = sum(k -> (d = R * y[k] + t - z[k]; dot(d, d)), 1:9)
        best = min(best, sqrt(s / 9))
    end
    @test fl.rmsd ≈ best rtol = 1e-12

    m = falses(9); m[1:4] .= true
    flm = search_floor(rset, spec, y, z; mask = m)
    @test flm.rmsd <= fl.rmsd + 1e-12
    @test_throws ArgumentError search_floor(rset, spec, y, z; mask = falses(9))
end


@testset "ligand_rmsd" begin
    T = Float64
    rng = MersenneTwister(7)
    a = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) for _ in 1:20]
    b = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) for _ in 1:20]
    direct = sqrt(sum(k -> sum(abs2, a[k] - b[k]), 1:20) / 20)
    @test ligand_rmsd(a, b) ≈ direct
    @test ligand_rmsd(a, a) == 0
    m = falses(20); m[3] = true; m[11] = true
    @test ligand_rmsd(a, b; mask = m) ≈
          sqrt((sum(abs2, a[3] - b[3]) + sum(abs2, a[11] - b[11])) / 2)
    @test_throws ArgumentError ligand_rmsd(a, b; mask = falses(20))
    @test_throws DimensionMismatch ligand_rmsd(a, b[1:19])

    d = SVector{3,T}(1, -2, 2)
    @test ligand_rmsd([p + d for p in a], a) ≈ 3.0
end


@testset "kabsch and superposition" begin
    T = Float64
    rng = MersenneTwister(11)
    P = [SVector{3,T}(randn(rng), randn(rng), randn(rng)) for _ in 1:12]
    q = SVector{4,T}(0.3, -0.5, 0.7, 0.41); q = q / norm(q)
    Rtrue = rotmatrix(q)
    ttrue = SVector{3,T}(4, -1, 2)
    Q = [Rtrue * p + ttrue for p in P]
    R, t = kabsch(P, Q)
    @test R ≈ Rtrue atol = 1e-12
    @test t ≈ ttrue atol = 1e-12
    @test LinearAlgebra.det(R) ≈ 1
    @test superposed_rmsd(P, Q) < 1e-12

    Qm = [SVector{3,T}(-p[1], p[2], p[3]) for p in P]
    Rm, _ = kabsch(P, Qm)
    @test LinearAlgebra.det(Rm) ≈ 1
    @test superposed_rmsd(P, Qm) > 0.5

    q2 = SVector{4,T}(-0.2, 0.9, 0.1, 0.33); q2 = q2 / norm(q2)
    R2 = rotmatrix(q2)
    Pm = [R2 * p + SVector{3,T}(7, 7, 7) for p in P]
    @test superposed_rmsd(Pm, Qm) ≈ superposed_rmsd(P, Qm) atol = 1e-10
    @test_throws ArgumentError kabsch(P[1:2], Q[1:2])
end


@testset "selections and residue grouping" begin
    at = read_atoms(VFIX; chains = "A", center = false)
    ids, nres = residue_ids(at)
    @test nres == 3
    @test length(ids) == 15
    @test all(count(==(r), ids) == 5 for r in 1:3)

    m = ca_mask(at)
    @test count(m) == 3
    @test all(at.atomname[i] == "CA" for i in findall(m))

    fake = joinpath(VDIR, "ca.pdb")
    open(fake, "w") do io
        println(io, vline(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"))
        println(io, vline(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"))
        println(io, "HETATM    3 CA    CA A   2       9.000   0.000   0.000" *
                    "  1.00  0.00          CA")
        println(io, "END")
    end
    atc = read_atoms(fake; center = false, keep_hetero = true)
    @test count(ca_mask(atc)) == 1
    @test atc.element[findfirst(ca_mask(atc))] == "C"
end


@testset "interface masks and interface RMSD" begin
    T = Float64
    rec = read_atoms(VFIX; T = T, chains = "A", center = false)
    lig = read_atoms(VFIX; T = T, chains = "B", center = false)
    native = native_ligand_coords(rec, lig)
    @test native ≈ lig.coords

    recmask, ligmask = interface_masks(rec, lig; cutoff = 10.0, backbone = true)

    @test count(recmask) == 4
    @test count(ligmask) == 4
    @test all(rec.resnum[i] == 1 for i in findall(recmask))
    @test all(lig.resnum[j] == 1 for j in findall(ligmask))
    @test !any(rec.atomname[i] == "CB" for i in findall(recmask))

    rm2, lm2 = interface_masks(rec, lig; cutoff = 10.0, backbone = false)
    @test count(rm2) == 5 && count(lm2) == 5

    rm3, _ = interface_masks(rec, lig; cutoff = 16.0)
    @test count(rm3) == 8

    @test interface_rmsd(rec, lig, native, native) < 1e-12

    shifted = [p + SVector{3,T}(3, 0, 0) for p in native]
    irm = interface_rmsd(rec, lig, native, shifted)
    @test 0 < irm < 3.0
    @test ligand_rmsd(shifted, native) ≈ 3.0

    far = [p + SVector{3,T}(500, 0, 0) for p in native]
    @test_throws ArgumentError interface_rmsd(rec, lig, far, far)
end


@testset "hit_rank and cluster centres" begin
    @test hit_rank([30.0, 21.0, 9.5, 2.0], 10.0) == 3
    @test hit_rank([30.0, 21.0, 19.5], 10.0) == 0
    @test hit_rank([10.0], 10.0) == 1

    N = 12
    D = fill(50.0, N, N)
    for i in 1:N, j in 1:N
        D[i, j] = (i - 1) ÷ 4 == (j - 1) ÷ 4 ? 2.0 : 50.0
    end
    for i in 1:N; D[i, i] = 0.0; end
    cl = greedy_cluster(D, 9.0)
    @test length(cl) == 3
    @test sort(vcat(cl...)) == 1:N
    for c in cl

        @test length(c) == 4
        @test all(D[c[1], m] <= 9.0 for m in c)
    end

    D2 = copy(D); for i in 1:N; D2[i, i] = 99.0; end
    cl2 = greedy_cluster(D2, 9.0)
    @test sort(vcat(cl2...)) == 1:N
end


@testset "pose_transforms feeds the clustering RMSD" begin

    T = Float64
    rng = MersenneTwister(3)
    spec = GridSpec{T}((32, 32, 32), T(1.2), SVector{3,T}(-19.2, -19.2, -19.2))
    y = [SVector{3,T}(5randn(rng), 5randn(rng), 5randn(rng)) for _ in 1:25]
    y .-= (sum(y) / 25,)
    rset = rotations_hopf(T, 4, 2)
    poses = [Pose{T}(-rand(rng), Int32(1 + (i - 1) % length(rset)),
                     Int16(mod(2i, 32)), Int16(mod(3i, 32)), Int16(mod(5i, 32)))
             for i in 1:8]
    Rs, ts = pose_transforms(poses, rset.rots, spec)
    D = Matrix{T}(undef, 8, 8)
    pairwise_rmsd!(D, Rs, ts, second_moment(y), 25)
    for i in 1:8, j in 1:8
        direct = ligand_rmsd(pose_coords(y, Rs[i], ts[i]), pose_coords(y, Rs[j], ts[j]))
        @test D[i, j] ≈ direct atol = 1e-6
    end
end

end

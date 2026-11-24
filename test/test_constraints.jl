using Test, JuPIPER, StaticArrays, LinearAlgebra, Printf


using BioStructures

include(joinpath(@__DIR__, "..", "scripts", "cdr.jl"))
include(joinpath(@__DIR__, "..", "scripts", "unbound.jl"))
include(joinpath(@__DIR__, "..", "scripts", "restraintfile.jl"))

const C97_STRUCT = joinpath(@__DIR__, "..", "data", "structures")
const C97_REST = joinpath(@__DIR__, "..", "data", "restraints", "1f3g_1poh_20noe.json")
c97_have(ids...) = all(id -> isfile(joinpath(C97_STRUCT, id * ".pdb")), ids)

const ONE2THREE = Dict('A' => "ALA", 'R' => "ARG", 'N' => "ASN", 'D' => "ASP",
                       'C' => "CYS", 'Q' => "GLN", 'E' => "GLU", 'G' => "GLY",
                       'H' => "HIS", 'I' => "ILE", 'L' => "LEU", 'K' => "LYS",
                       'M' => "MET", 'F' => "PHE", 'P' => "PRO", 'S' => "SER",
                       'T' => "THR", 'W' => "TRP", 'Y' => "TYR", 'V' => "VAL")

"""One C-alpha per residue, so `chain_residues` and `residue_mask` see a real table."""
function fake_table(chains::Vector{Pair{String,String}}; T = Float64)
    co = SVector{3,T}[]; el = String[]; an = String[]; rn = String[]
    nu = Int32[]; ic = Char[]; ch = String[]
    for (cid, seq) in chains, (k, aa) in enumerate(seq)
        push!(co, SVector{3,T}(k, 0, 0)); push!(el, "C"); push!(an, "CA")
        push!(rn, ONE2THREE[aa]); push!(nu, Int32(k)); push!(ic, ' '); push!(ch, cid)
    end
    n = length(co)
    ids = unique(ch)
    rep = JuPIPER.IngestReport("synthetic", 1, 1, ids, ids, n, n, 0, 0, 0,
                               Dict{String,Int}(), 0, Dict{String,Int}(), 0, 0,
                               n, 0, 0)
    return AtomTable{T}(co, el, an, rn, nu, ic, ch, ones(T, n), zeros(T, n),
                        falses(n), falses(n), zero(SVector{3,T}), false, rep)
end

"""
A synthetic light chain with CDR-L1 at 24-34, L2 at 50-56, L3 at 89-97, built
so that every published anchor holds and nothing else in the chain can be
mistaken for one: no Cys outside 23 and 88, no Trp at all, no Phe-Gly-x-Gly
except after L3.
"""
function synth_light()
    s = fill('A', 110)
    s[23] = 'C'
    s[24:34] .= 'S'
    s[35] = 'W'; s[36] = 'Y'; s[37] = 'Q'
    s[48] = 'I'; s[49] = 'Y'
    s[50:56] .= 'T'
    s[88] = 'C'
    s[89:97] .= 'V'
    s[98] = 'F'; s[99] = 'G'; s[100] = 'P'; s[101] = 'G'
    return String(s)
end

"""
A synthetic heavy chain with CDR-H1 at 26-35, H2 at 50-65, H3 at 99-106.
"""
function synth_heavy()
    s = fill('A', 115)
    s[22] = 'C'
    s[26:35] .= 'S'
    s[36] = 'W'; s[37] = 'V'
    s[45] = 'L'; s[46] = 'E'; s[47] = 'W'; s[48] = 'I'; s[49] = 'G'
    s[50:65] .= 'T'
    s[66] = 'K'; s[67] = 'A'; s[68] = 'S'
    s[96] = 'C'; s[97] = 'A'; s[98] = 'R'
    s[99:106] .= 'V'
    s[107] = 'W'; s[108] = 'G'; s[109] = 'Q'; s[110] = 'G'
    return String(s)
end

@testset "constraints" begin


@testset "CDR annotation" begin
    L = synth_light(); H = synth_heavy()

    l1, l2, l3 = light_cdrs(L)
    @test l1 == 24:34
    @test l2 == 50:56
    @test l3 == 89:97
    h1, h2, h3 = heavy_cdrs(H)
    @test h1 == 26:35
    @test h2 == 50:65
    @test h3 == 99:106

    @test first(l2) == last(l1) + 16
    @test length(l2) == 7
    @test first(l3) == last(l2) + 33
    @test first(h2) == last(h1) + 15
    @test L[first(l1) - 1] == 'C'
    @test L[last(l1) + 1] == 'W'
    @test L[first(l3) - 1] == 'C'
    @test H[first(h1) - 4] == 'C'
    @test H[last(h1) + 1] == 'W'
    @test H[first(h3) - 3] == 'C'
    @test 10 <= length(l1) <= 17
    @test 7 <= length(l3) <= 11
    @test 10 <= length(h1) <= 12
    @test 16 <= length(h2) <= 19

    noW = collect(L); noW[35] = 'A'
    @test_throws ErrorException light_cdrs(String(noW))
    noC = collect(L); noC[88] = 'A'
    @test_throws ErrorException light_cdrs(String(noC))
    noC1 = collect(L); noC1[23] = 'A'
    @test_throws ErrorException light_cdrs(String(noC1))
    noFG = collect(L); noFG[98] = 'A'
    @test_throws ErrorException light_cdrs(String(noFG))
    hnoC = collect(H); hnoC[22] = 'A'
    @test_throws ErrorException heavy_cdrs(String(hnoC))
    hnoW = collect(H); hnoW[36] = 'A'
    @test_throws ErrorException heavy_cdrs(String(hnoW))
    hnoK = collect(H); hnoK[66:68] .= 'P'
    @test_throws ErrorException heavy_cdrs(String(hnoK))
    hnoWG = collect(H); hnoWG[107] = 'A'
    @test_throws ErrorException heavy_cdrs(String(hnoWG))

    at = fake_table(["H" => H, "L" => L])
    cdr, loops = fab_cdr_residues(at; heavy = "H", light = "L")
    @test length(cdr) == length(l1) + length(l2) + length(l3) +
                         length(h1) + length(h2) + length(h3)
    @test length(cdr) == 61
    sel = noncdr_selection(at, cdr)
    @test count(sel) == length(at) - length(cdr)
    @test count(.!sel) == length(cdr)
    @test ("H", 99) in cdr && !(("H", 98) in cdr)
    @test ("L", 34) in cdr && !(("L", 35) in cdr)

    @test last(l1) < first(l2) && last(l2) < first(l3)
    @test last(h1) < first(h2) && last(h2) < first(h3)

    m = spatial_mask(at, sel; repulsion = 2.0)
    @test masked_residue_count(at, m) == length(at) - length(cdr)
    @test !is_identity(m)
    @test is_identity(spatial_mask(at, sel; repulsion = 1.0, attraction = 1.0))

    @test_throws ArgumentError chain_residues(at, "Z")

    if c97_have("1FGN")
        fab = read_atoms(joinpath(C97_STRUCT, "1FGN.pdb"); T = Float64,
                         chains = ["H", "L"])
        rcdr, rloops = fab_cdr_residues(fab; heavy = "H", light = "L")

        @test (rloops.L1[1][1], rloops.L1[end][1]) == (24, 34)
        @test (rloops.L2[1][1], rloops.L2[end][1]) == (50, 56)
        @test (rloops.L3[1][1], rloops.L3[end][1]) == (89, 97)
        @test (rloops.H1[1][1], rloops.H1[end][1]) == (26, 35)
        @test (rloops.H2[1][1], rloops.H2[end][1]) == (50, 66)
        @test (rloops.H3[1][1], rloops.H3[end][1]) == (99, 106)
        @test length(rcdr) == 62

        _, hseq = chain_residues(fab, "H")
        _, lseq = chain_residues(fab, "L")
        @test lseq[23] == 'C' && lseq[35] == 'W' && lseq[36] == 'Y'
        @test lseq[48] == 'I' && lseq[49] == 'Y' && lseq[88] == 'C'
        @test lseq[98] == 'F' && lseq[99] == 'G' && lseq[101] == 'G'
        @test hseq[22] == 'C' && hseq[36] == 'W' && hseq[37] == 'V'
        @test hseq[45:49] == "LEWIG"
        @test hseq[96] == 'C' && hseq[97] == 'A' && hseq[98] == 'R'
        @test hseq[107] == 'W' && hseq[108] == 'G' && hseq[110] == 'G'

        rsel = noncdr_selection(fab, rcdr)
        @test count(rsel) == 2786
        @test masked_residue_count(fab, spatial_mask(fab, rsel; repulsion = 2)) == 366
    else
        @info "1FGN not fetched -- skipping the real-Fab CDR testset. " *
              "bash scripts/fetch_structures.sh 1FGN"
    end
end


@testset "restraint transcription" begin
    @test isfile(C97_REST)
    groups, meta = load_restraint_groups(C97_REST)

    @test length(groups) == 20
    @test meta.n == 20
    @test sum(length(g.rec) * length(g.lig) for g in groups) == 245

    @test count(g -> length(g.rec) == 1, groups) == 11
    @test count(g -> length(g.lig) == 1, groups) == 9

    e2a = sort(unique(reduce(vcat, [g.rec for g in groups])))
    hpr = sort(unique(reduce(vcat, [g.lig for g in groups])))
    @test e2a == [38, 39, 40, 45, 46, 68, 69, 71, 72, 78, 80, 86, 94, 96, 141]
    @test hpr == [12, 15, 16, 17, 20, 48, 51, 52, 54, 56]

    @test all(g -> g.dmin == 0.0, groups)
    @test all(g -> g.dmax == 4.5, groups)

    @test all(g -> g.cutoff == 0, groups)

    txt = read(C97_REST, String)
    @test occursin("cluspro.org/examples/ja026939x_s2.txt", txt)
    @test occursin("cluspro.org/examples/ja026939x_s2.json", txt)
    @test occursin("Nature Protocols", txt)
    for m in eachmatch(r"\"source_\w*_?sha256\":\s*\"([0-9a-f]*)\"", txt)
        d = m.captures[1]
        @test length(d) == 64

        @test d != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    end

    @test occursin(r"\"set_cutoff\":\s*\{\"value\":\s*null", txt)

    if c97_have("1F3G", "1POH")
        rec = read_atoms(joinpath(C97_STRUCT, "1F3G.pdb"); T = Float64, chains = "A")
        lig = read_atoms(joinpath(C97_STRUCT, "1POH.pdb"); T = Float64, chains = "A")
        set = build_restraint_set(groups, rec, lig, 0)
        @test length(set) == 20
        @test nrestraints(set) == 245
        @test set.cutoff == 0
        @test all(g -> g.cutoff == 0, set.groups)

        @test restraint_set(set.groups; cutoff = 19).cutoff == 19

        @test_throws ArgumentError restraint_set(set.groups; cutoff = 20)

        bad = [(rec = [9999], lig = groups[1].lig, dmin = 0.0, dmax = 4.5, cutoff = 0)]
        @test_throws ErrorException build_restraint_set(bad, rec, lig, 0)
    else
        @info "1F3G/1POH not fetched -- skipping restraint set construction. " *
              "bash scripts/fetch_structures.sh \"1F3G 1POH\""
    end
end


@testset "unbound native reference" begin

    mk(off, names) = Dict(i + off => (SVector{3,Float64}(i, 0, 0), names[i])
                          for i in eachindex(names))
    names = [ONE2THREE[c] for c in "ACDEFGHIKLMNPQRSTVWY"]
    a = mk(0, names); b = mk(300, names)
    delta, votes = numbering_offset(a, b)
    @test delta == 300
    @test votes == length(names)
    @test numbering_offset(a, a)[1] == 0
    @test_throws ErrorException numbering_offset(a, Dict{Int,Tuple{SVector{3,Float64},String}}())

    if c97_have("1POH", "1GGR", "1F3G")
        poh = read_atoms(joinpath(C97_STRUCT, "1POH.pdb"); T = Float64, chains = "A")
        ggr = read_atoms(joinpath(C97_STRUCT, "1GGR.pdb"); T = Float64)
        d1 = ca_by_residue(poh, "A"); d2 = ca_by_residue(ggr, "B")
        @test !isempty(d1) && !isempty(d2)

        @test length(intersect(keys(d1), keys(d2))) == 0
        off, nv = numbering_offset(d1, d2)
        @test off == 300
        @test nv >= 80

        nrec = read_atoms(joinpath(C97_STRUCT, "1GGR.pdb"); T = Float64, chains = "A")
        nlig = read_atoms(joinpath(C97_STRUCT, "1GGR.pdb"); T = Float64, chains = "B")
        want = native_ligand_coords(nrec, nlig)
        got, diag = unbound_native(nrec, nlig, ggr, ["A" => "A"], ["B" => "B"];
                                   verbose = false)
        @test length(got) == length(want)
        @test maximum(norm(got[i] - want[i]) for i in eachindex(got)) < 1e-8
        @test diag.lig_unbound_rmsd < 1e-8
        @test diag.rec_rmsd < 1e-8
        @test diag.lig_matched == count(ca_mask(nlig))

        f3g = read_atoms(joinpath(C97_STRUCT, "1F3G.pdb"); T = Float64, chains = "A")
        nat, d = unbound_native(f3g, poh, ggr, ["A" => "A"], ["A" => "B"];
                                verbose = false)
        @test length(nat) == length(poh)

        @test 0 < d.lig_unbound_rmsd < 3.0
        @test 0 < d.rec_rmsd < 3.0

        @test_throws ErrorException superpose_chains(f3g, ggr, ["A" => "B"];
                                                     verbose = false)
    else
        @info "1POH/1GGR/1F3G not fetched -- skipping the unbound-reference " *
              "testset. bash scripts/fetch_structures.sh \"1F3G 1POH 1GGR\""
    end
end

end

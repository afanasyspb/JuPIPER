using Test, JuPIPER, StaticArrays, LinearAlgebra, Printf
using JuPIPER: AtomTable, IngestReport, PDBSource, BioStructuresPDB,
               ACP_STANDARD_ATOM_COUNT, PROTEIN_RESNAMES, MODIFIED_RESIDUES,
               nresidues, chargemodel_name


using BioStructures


"""
Format one ATOM/HETATM record in fixed-column PDB v3.

Column arithmetic is done once, here, rather than by counting spaces in a string
literal 60 times. The atom-name justification rule is real and load-bearing:
a *two-letter* element occupies columns 13-14, a one-letter element starts at
column 14. That is exactly what separates ` CA ` (alpha carbon) from `CA  `
(calcium), and it is why [`infer_element`](@ref) needs the residue name.
"""
function pdbline(serial, name, altloc, resname, chain, resnum, icode,
                 x, y, z, occ, b, elem; hetatm::Bool = false)
    rec = hetatm ? "HETATM" : "ATOM  "
    nm = length(name) >= 4 ? name :
         (length(strip(elem)) == 2 ? rpad(name, 4) : rpad(" " * name, 4))
    return string(rec, lpad(serial, 5), " ", nm, altloc, rpad(resname, 3), " ",
                  chain, lpad(resnum, 4), icode, "   ",
                  @sprintf("%8.3f%8.3f%8.3f", x, y, z),
                  @sprintf("%6.2f%6.2f", occ, b),
                  " "^10, lpad(elem, 2))
end

"Residue definitions for the fixture: (resname, [(atomname, element)...])."
const FIXTURE_A = [

    ("GLY", 1, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("HA2", "H")]),

    ("ALA", 2, [("N", "N"), ("CA", ""), ("C", "C"), ("O", "O"), ("CB", "C")]),
    ("ARG", 3, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                ("CG", "C"), ("CD", "C"), ("NE", "N"), ("CZ", "C"),
                ("NH1", "N"), ("NH2", "N")]),

    ("ASP", 4, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                ("CG", "C"), ("OD1", "O"), ("OD2", "O")]),

    ("TYR", 5, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                ("CG", "C"), ("CD1", "C"), ("CD2", "C"), ("CE1", "C"),
                ("CE2", "C"), ("CZ", "C"), ("OH", "O"), ("OXT", "O")]),
]

const FIXTURE_B = [

    ("MSE", 1, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                ("CG", "C"), ("SE", "SE"), ("CE", "C")]),

    ("LYS", 2, [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                ("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")]),
]

"Waters, a sulfate and a calcium ion -- all HETATM, none of them protein."
const FIXTURE_W = [
    ("HOH", 101, [("O", "O")]),
    ("HOH", 102, [("O", "O")]),
    ("SO4", 201, [("S", "S"), ("O1", "O"), ("O2", "O"), ("O3", "O"), ("O4", "O")]),

    ("CA", 301, [("CA", "")]),
]

function write_fixture(path)
    open(path, "w") do io
        serial = 0
        xyz(i) = (1.5i, 0.75i + 2.0, -0.5i + 1.0)
        emit(io, resn, num, ch, atoms; het = false) = begin
            for (an, el) in atoms
                if resn == "ASP" && an == "OD1"

                    for (alt, occ) in (('A', 0.70), ('B', 0.30))
                        serial += 1
                        x, y, z = xyz(serial)
                        println(io, pdbline(serial, an, alt, resn, ch, num, ' ',
                                            x, y, z, occ, 15.0, el; hetatm = het))
                    end
                else
                    serial += 1
                    x, y, z = xyz(serial)
                    println(io, pdbline(serial, an, ' ', resn, ch, num, ' ',
                                        x, y, z, 1.00, 15.0, el; hetatm = het))
                end
            end
        end

        println(io, "MODEL        1")
        for (resn, num, atoms) in FIXTURE_A
            emit(io, resn, num, "A", atoms)
        end
        println(io, "TER")
        for (resn, num, atoms) in FIXTURE_B
            emit(io, resn, num, "B", atoms; het = (resn in keys(MODIFIED_RESIDUES)))
        end
        println(io, "TER")
        for (resn, num, atoms) in FIXTURE_W
            emit(io, resn, num, "W", atoms; het = true)
        end
        println(io, "ENDMDL")

        println(io, "MODEL        2")
        emit(io, "GLY", 1, "A", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")])
        println(io, "ENDMDL")
        println(io, "END")
    end
    return path
end

const FIXTURE_DIR = mktempdir()
const FIXTURE = write_fixture(joinpath(FIXTURE_DIR, "fixture.pdb"))


const N_CHAIN_A = 4 + 5 + 11 + 8 + 13
const N_CHAIN_B = 8 + 9
const N_KEPT = N_CHAIN_A + N_CHAIN_B

@testset "ACP 18-type table" begin
    tbl = load_acp_types()
    pot = load_dars()

    @test tbl.labels == ACP18_LABELS
    @test pot.labels == ACP18_LABELS
    @test length(ACP18_LABELS) == 18
    @test length(unique(ACP18_LABELS)) == 18

    std = setdiff(keys(tbl.byres), keys(MODIFIED_RESIDUES))
    @test length(std) == 20
    @test sum(length(tbl.byres[r]) for r in std) == ACP_STANDARD_ATOM_COUNT
    @test ACP_STANDARD_ATOM_COUNT == 167

    want = Dict("GLY" => 4, "ALA" => 5, "SER" => 6, "CYS" => 6, "THR" => 7,
                "VAL" => 7, "PRO" => 7, "LEU" => 8, "ILE" => 8, "ASN" => 8,
                "ASP" => 8, "MET" => 8, "GLN" => 9, "GLU" => 9, "LYS" => 9,
                "HIS" => 10, "ARG" => 11, "PHE" => 11, "TYR" => 12, "TRP" => 14)
    for (r, n) in want
        @test length(tbl.byres[r]) == n
    end
    @test sum(values(want)) == ACP_STANDARD_ATOM_COUNT

    used = Set(v for sub in values(tbl.byres) for v in values(sub))
    union!(used, values(tbl.anyres))
    @test length(used) == 18

    @test acp_type(tbl, "ALA", "N")  == findfirst(==("N"), ACP18_LABELS)
    @test acp_type(tbl, "ALA", "CA") == findfirst(==("Ca"), ACP18_LABELS)
    @test acp_type(tbl, "ALA", "C")  == findfirst(==("C"), ACP18_LABELS)
    @test acp_type(tbl, "ALA", "O")  == findfirst(==("O"), ACP18_LABELS)

    @test acp_type(tbl, "GLY", "CA") == findfirst(==("GCa"), ACP18_LABELS)
    @test acp_type(tbl, "GLY", "CA") != acp_type(tbl, "ALA", "CA")

    dod = findfirst(==("DOd"), ACP18_LABELS)
    @test acp_type(tbl, "ASP", "OD1") == dod
    @test acp_type(tbl, "GLU", "OE2") == dod
    @test acp_type(tbl, "TYR", "CZ")  == findfirst(==("YCz"), ACP18_LABELS)
    @test acp_type(tbl, "PHE", "CZ")  == findfirst(==("FCz"), ACP18_LABELS)
    @test acp_type(tbl, "TYR", "CZ")  != acp_type(tbl, "PHE", "CZ")
    @test acp_type(tbl, "LYS", "NZ")  == findfirst(==("KNz"), ACP18_LABELS)
    @test acp_type(tbl, "ARG", "NE")  == findfirst(==("RNe"), ACP18_LABELS)
    @test acp_type(tbl, "ARG", "NH1") == acp_type(tbl, "ARG", "NH2")
    @test acp_type(tbl, "CYS", "SG")  == findfirst(==("CSg"), ACP18_LABELS)
    @test acp_type(tbl, "MET", "SD")  == acp_type(tbl, "CYS", "SG")
    @test acp_type(tbl, "MSE", "SE")  == acp_type(tbl, "CYS", "SG")
    @test acp_type(tbl, "SER", "OG")  == acp_type(tbl, "THR", "OG1")
    @test acp_type(tbl, "TYR", "OH")  == acp_type(tbl, "SER", "OG")

    @test acp_type(tbl, "ASN", "OD1") == acp_type(tbl, "GLY", "O")
    @test acp_type(tbl, "ASN", "CG")  == acp_type(tbl, "GLY", "C")

    @test acp_type(tbl, "TYR", "OXT") == dod
    @test acp_type(tbl, "GLY", "OXT") == dod
    @test isempty(intersect(keys(tbl.anyres), ("N", "CA", "C", "O")))

    @test acp_type(tbl, "ALA", "NOPE") == 0
    @test acp_type(tbl, "XYZ", "CA") == 0
    @test nresidues(tbl) == 21

    bad = joinpath(FIXTURE_DIR, "bad.tsv")
    write(bad, "ALA\tCB\tNotAType\n")
    @test_throws ArgumentError load_acp_types(bad)
    write(bad, "ALA\tCB\tCb\nALA\tCB\tCb\n")
    @test_throws ArgumentError load_acp_types(bad)
end

@testset "element inference" begin

    @test infer_element("ALA", "CA") == "C"
    @test infer_element("GLY", "CA") == "C"
    @test infer_element("MET", "SD") == "S"
    @test infer_element("ARG", "NH1") == "N"
    @test infer_element("TYR", "OH") == "O"
    @test infer_element("MSE", "SE") == "SE"
    @test infer_element("MET", "CE") == "C"
    @test infer_element("LYS", "NZ") == "N"

    @test infer_element("HEM", "FE") == "FE"
    @test infer_element("ZN", "ZN") == "ZN"
    @test infer_element("NAG", "CL") == "CL"

    @test infer_element("CA", "CA") == "CA"
    @test infer_element("CD", "CD") == "CD"

    @test infer_element("SEP", "CA") == "C"
    @test infer_element("TPO", "CA") == "C"
    @test infer_element("PTR", "CD1") == "C"
    @test infer_element("UNK", "NE2") == "N"

    @test infer_element("ALA", "1HB") == "H"
    @test infer_element("LEU", "2HG1") == "H"
    @test infer_element("ALA", "") == ""
    @test infer_element("ALA", "   ") == ""

    prefixes = Set(["N", "CA", "CB", "CG", "CD", "CE", "CZ", "CH", "OD", "OE",
                    "OG", "OH", "ND", "NE", "NH", "NZ", "SD", "SG", "C", "O"])
    @test JuPIPER.AMBIGUOUS_TWO_LETTER ==
          intersect(JuPIPER.TWO_LETTER_ELEMENTS, prefixes)
end

@testset "PDB ingestion" begin
    at = read_atoms(FIXTURE)
    r = at.report

    @test at isa AtomTable{Float64}
    @test length(at) == N_KEPT
    @test JuPIPER.natoms(at) == N_KEPT

    @test r.models_present == 2
    @test r.model_selected == 1

    @test r.n_hydrogen == 1
    @test r.n_water == 2
    @test r.n_hetero_dropped == 6
    @test r.hetero_resnames == Dict("SO4" => 5, "CA" => 1)
    @test r.n_modified_kept == 1
    @test r.modified_resnames == Dict("MSE" => 1)
    @test r.n_altloc == 1
    @test r.n_element_inferred == 1
    @test r.n_residues == 7
    @test r.natoms_seen == r.natoms_kept + r.n_hydrogen + r.n_water +
                           r.n_hetero_dropped
    @test r.natoms_kept == N_KEPT

    @test Set(r.chains_present) == Set(["A", "B", "W"])
    @test Set(r.chains_kept) == Set(["A", "B"])

    @test "MSE" in at.resname
    @test "SE" in at.atomname
    @test at.element[findfirst(==("SE"), at.atomname)] == "SE"

    ala_ca = findfirst(i -> at.resname[i] == "ALA" && at.atomname[i] == "CA",
                       eachindex(at.resname))
    @test at.element[ala_ca] == "C"

    od1 = findfirst(i -> at.resname[i] == "ASP" && at.atomname[i] == "OD1",
                    eachindex(at.resname))
    @test od1 !== nothing
    @test at.occupancy[od1] ≈ 0.70
    @test count(i -> at.resname[i] == "ASP" && at.atomname[i] == "OD1",
                eachindex(at.resname)) == 1

    @test r.n_cterm_complete == 1
    @test r.n_cterm_no_oxt == 1
    @test all(at.resname[i] == "GLY" || at.resname[i] == "MSE"
              for i in eachindex(at.is_nterm) if at.is_nterm[i])
    @test all(at.resname[i] == "TYR" for i in eachindex(at.is_cterm) if at.is_cterm[i])
    @test count(at.is_cterm) == 13
    @test !any(i -> at.resname[i] == "LYS" && at.is_cterm[i], eachindex(at.is_cterm))

    @test at.centred
    @test norm(sum(at.coords) / length(at)) < 1e-10
    raw = read_atoms(FIXTURE; center = false)
    @test !raw.centred
    @test raw.centroid == zero(SVector{3,Float64})
    @test norm(sum(raw.coords) / length(raw) - at.centroid) < 1e-10
    back = uncenter!(deepcopy(at))
    @test maximum(norm.(back.coords .- raw.coords)) < 1e-10

    c0 = at.centroid
    center!(at)
    @test at.centroid ≈ c0

    @test all(>(0), extent(at))

    @test length(read_atoms(FIXTURE; chains = "A")) == N_CHAIN_A
    @test length(read_atoms(FIXTURE; chains = ["A", "B"])) == N_KEPT
    @test length(read_atoms(FIXTURE; chains = "B")) == N_CHAIN_B
    @test length(read_atoms(FIXTURE; keep_hydrogens = true)) == N_KEPT + 1
    @test length(read_atoms(FIXTURE; keep_water = true)) == N_KEPT + 2
    @test length(read_atoms(FIXTURE; keep_hetero = true)) == N_KEPT + 6
    @test length(read_atoms(FIXTURE; model = 2)) == 4
    @test_throws ArgumentError read_atoms(FIXTURE; model = 7)

    het = read_atoms(FIXTURE; keep_hetero = true)
    ion = findfirst(i -> het.resname[i] == "CA", eachindex(het.resname))
    @test het.element[ion] == "CA"

    at32 = read_atoms(FIXTURE; T = Float32)
    @test at32 isa AtomTable{Float32}
    @test eltype(at32) === Float32
    @test length(at32) == N_KEPT

    @test occursin("atoms kept", sprint(show, r))
    @test occursin("AtomTable", sprint(show, at))
end

@testset "reader seam" begin

    in_base = [m for m in methods(JuPIPER._read_atoms) if m.module === JuPIPER]
    @test length(in_base) == 1
    @test in_base[1].sig.parameters[2] === PDBSource
    @test !any(m -> m.module === JuPIPER && m.sig.parameters[2] === BioStructuresPDB,
               methods(JuPIPER._read_atoms))

    @test any(m -> m.module !== JuPIPER && m.sig.parameters[2] === BioStructuresPDB,
              methods(JuPIPER._read_atoms))

    struct UnservedSource <: PDBSource end
    @test_throws ArgumentError read_atoms(FIXTURE; source = UnservedSource())
end

@testset "parametrisation" begin
    tbl = load_acp_types()
    at = read_atoms(FIXTURE)

    par = parametrize(tbl, at)
    @test length(par) == length(at)

    @test isempty(par.untyped)
    @test isempty(par.unknown_element)
    @test all(t -> 1 <= t <= 18, par.acptype)
    @test par.labels == ACP18_LABELS

    for i in eachindex(par.radius)
        @test par.radius[i] ≈ vdw_radius(Float64, at.element[i])
    end
    @test par.radius[findfirst(==("SE"), at.atomname)] ≈ 1.90
    @test has_vdw_radius("SE") && has_vdw_radius("C") && !has_vdw_radius("XX")

    h = type_histogram(par)
    @test sum(values(h)) == length(at)
    @test h["GCa"] == 1
    @test h["CSg"] == 1

    @test par.chargemodel === :zero
    @test all(iszero, par.charge)
    @test par.net_charge == 0

    fq = parametrize(tbl, at; charges = FormalCharges())
    @test fq.chargemodel === :formal
    idx(rn, an) = findfirst(i -> at.resname[i] == rn && at.atomname[i] == an,
                            eachindex(at.resname))
    @test fq.charge[idx("ASP", "OD1")] ≈ -0.5
    @test fq.charge[idx("ASP", "OD2")] ≈ -0.5
    @test fq.charge[idx("LYS", "NZ")] ≈ 1.0
    @test fq.charge[idx("ARG", "NE")] ≈ 1 / 3
    @test fq.charge[idx("ARG", "NH1")] ≈ 1 / 3
    @test fq.charge[idx("ARG", "CZ")] == 0
    @test fq.charge[idx("TYR", "OH")] == 0

    @test fq.charge[idx("GLY", "N")] ≈ 1.0
    @test fq.charge[idx("MSE", "N")] ≈ 1.0

    @test fq.charge[idx("TYR", "OXT")] ≈ -0.5
    @test fq.charge[idx("TYR", "O")] ≈ -0.5

    @test fq.charge[idx("LYS", "O")] == 0

    @test fq.net_charge ≈ 2.0
    @test sum(fq.charge) ≈ fq.net_charge

    @test atom_charge(ZeroCharges(), Float64, "ASP", "OD1", false, false) == 0
    @test atom_charge(FormalCharges(), Float64, "GLU", "OE1", false, false) ≈ -0.5
    @test atom_charge(FormalCharges(), Float64, "HIS", "ND1", false, false) == 0

    @test atom_charge(FormalCharges(), Float64, "ASP", "OXT", false, true) ≈ -0.5

    dod = findfirst(==("DOd"), ACP18_LABELS)
    obk = findfirst(==("O"), ACP18_LABELS)
    @test par.acptype[idx("TYR", "O")] == obk
    tc = parametrize(tbl, at; terminal_carboxylate = true)
    @test tc.acptype[idx("TYR", "O")] == dod
    @test tc.acptype[idx("TYR", "OXT")] == dod
    @test tc.acptype[idx("LYS", "O")] == obk
    @test tc.acptype[idx("GLY", "O")] == obk
    @test count(!=(0), tc.acptype .- par.acptype) == 1

    dirty = deepcopy(at)
    dirty.atomname[1] = "ZZZ"
    @test_logs (:warn,) parametrize(tbl, dirty)
    @test_throws ArgumentError parametrize(tbl, dirty; on_untyped = :error)
    quiet = parametrize(tbl, dirty; on_untyped = :ignore)
    @test quiet.untyped == Int32[1]
    @test quiet.acptype[1] == 0
    @test_throws ArgumentError parametrize(tbl, dirty; on_untyped = :nonsense)

    @test occursin("Parametrization", sprint(show, par))
    @test occursin("UNTYPED", sprint(show, quiet))
end

@testset "piper_structure bridge" begin
    tbl = load_acp_types()
    pot = load_dars()
    at = read_atoms(FIXTURE)
    par = parametrize(tbl, at; charges = FormalCharges())

    for P in (2, 4)
        s = piper_structure(at, par, pot; P = P)
        @test s isa Structure
        @test size(s.termw) == (length(at), 2 + P)
        @test length(s.coords) == length(at)

        @test all(isone, @view s.termw[:, 1])
        @test (@view s.termw[:, 2]) == par.charge
        for i in 1:length(at), p in 1:P
            @test s.termw[i, 2+p] ≈ pot.U[par.acptype[i], p]
        end
    end

    @test_throws ArgumentError piper_structure(at, par, pot; P = 0)
    @test_throws ArgumentError piper_structure(at, par, pot; P = 19)

    @test_throws ArgumentError piper_structure(read_atoms(FIXTURE; center = false),
                                               par, pot)

    dirty = deepcopy(at)
    dirty.atomname[1] = "ZZZ"
    dpar = parametrize(tbl, dirty; on_untyped = :ignore)
    @test_throws ArgumentError piper_structure(dirty, dpar, pot)
end

@testset "published scoring weights" begin

    @test PIPER_WEIGHTS_ENZYME_INHIBITOR.w1 == 2.1
    @test PIPER_WEIGHTS_ENZYME_INHIBITOR.w2 == 133.0
    @test PIPER_WEIGHTS_ENZYME_INHIBITOR.w3 == 2.2
    @test PIPER_WEIGHTS_ANTIBODY_ANTIGEN.w1 == 2.0
    @test PIPER_WEIGHTS_ANTIBODY_ANTIGEN.w2 == 400.0
    @test PIPER_WEIGHTS_ANTIBODY_ANTIGEN.w3 == 1.0
    w32 = PiperWeights{Float32}(PIPER_WEIGHTS_ENZYME_INHIBITOR)
    @test w32 isa PiperWeights{Float32}
    @test w32.w1 ≈ 2.1f0

    @test JuPIPER.DEFAULT_W_REP == PIPER_WEIGHTS_ENZYME_INHIBITOR.w1
    @test sum(values(JuPIPER.DARS_ACP_MIX)) ≈ 0.875
end

@testset "typing allocates nothing" begin

    function alloc_types(out, tbl, rn, an)
        acp_types!(out, tbl, rn, an)
        return @allocated acp_types!(out, tbl, rn, an)
    end
    function alloc_center(at)
        center!(at)
        return @allocated center!(at)
    end
    function alloc_extent(at)
        extent(at)
        return @allocated extent(at)
    end
    function alloc_lookup(tbl, r, a)
        acp_type(tbl, r, a)
        return @allocated acp_type(tbl, r, a)
    end

    tbl = load_acp_types()
    at = read_atoms(FIXTURE)
    out = Vector{Int8}(undef, length(at))

    @test alloc_types(out, tbl, at.resname, at.atomname) == 0
    @test alloc_lookup(tbl, "ARG", "NH1") == 0
    @test alloc_lookup(tbl, "TYR", "OXT") == 0
    @test alloc_center(at) == 0
    @test alloc_extent(at) == 0

    @test out == [acp_type(tbl, at.resname[i], at.atomname[i]) for i in eachindex(out)]
    @test_throws DimensionMismatch acp_types!(Vector{Int8}(undef, 3), tbl,
                                              at.resname, at.atomname)
end

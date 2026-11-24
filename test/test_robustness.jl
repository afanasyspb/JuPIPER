using Test, JuPIPER, StaticArrays, Printf

using BioStructures

const ROB_STRUCT = joinpath(@__DIR__, "..", "data", "structures")
rob_have(id) = isfile(joinpath(ROB_STRUCT, id * ".pdb"))

"Minimal fixed-column ATOM/HETATM record; see test_pdb.jl for the column rules."
function robline(serial, name, resname, chain, resnum, x, y, z, elem;
                 hetatm::Bool = false)
    rec = hetatm ? "HETATM" : "ATOM  "
    nm = length(strip(elem)) == 2 ? rpad(name, 4) : rpad(" " * name, 4)
    return string(rec, lpad(serial, 5), " ", nm, " ", rpad(resname, 3), " ",
                  chain, lpad(resnum, 4), " ", "   ",
                  @sprintf("%8.3f%8.3f%8.3f", x, y, z),
                  @sprintf("%6.2f%6.2f", 1.0, 0.0), " "^10, lpad(elem, 2))
end

"""
A globular core of `ncore` atoms in a ball of radius `rcore`, plus `ntail` atoms
strung out to `reach` -- the "tail that basically floats off into space".
`ntail = 0` gives the control.
"""
function tail_pdb(path; ncore = 400, rcore = 12.0, ntail = 0, reach = 80.0)
    open(path, "w") do io
        s = 0

        for i in 1:ncore
            u = (i * 0.6180339887498949) % 1.0
            v = (i * 0.4142135623730951) % 1.0
            w = (i * 0.7320508075688772) % 1.0
            r = rcore * cbrt(w)
            ct = 2u - 1
            st = sqrt(max(0.0, 1 - ct^2))
            ph = 2pi * v
            s += 1
            println(io, robline(s, "CA", "ALA", "A", i, r * st * cos(ph),
                                r * st * sin(ph), r * ct, "C"))
        end
        for j in 1:ntail
            s += 1
            println(io, robline(s, "CA", "GLY", "A", ncore + j,
                                reach * j / max(ntail, 1), 0.0, 0.0, "C"))
        end
        println(io, "END")
    end
    return path
end

@testset "robustness on dirty input" begin


@testset "Table 5: HETATM and nonstandard residues are classified, not fatal" begin
    dir = mktempdir()
    path = joinpath(dir, "dirty.pdb")
    open(path, "w") do io
        s = 0
        for (i, (rn, nm, el)) in enumerate([("ALA", "N", "N"), ("ALA", "CA", "C"),
                                            ("ALA", "C", "C"), ("ALA", "O", "O"),
                                            ("ALA", "CB", "C")])
            s += 1
            println(io, robline(s, nm, rn, "A", 1, i * 1.5, 0.0, 0.0, el))
        end

        for (i, (nm, el)) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"),
                                        ("O", "O"), ("SE", "SE")])
            s += 1
            println(io, robline(s, nm, "MSE", "A", 2, 8.0 + i * 1.5, 1.0, 0.0, el;
                                hetatm = true))
        end

        s += 1; println(io, robline(s, "O", "HOH", "A", 900, 3.0, 9.0, 0.0, "O"; hetatm = true))
        s += 1; println(io, robline(s, "CA", "CA", "A", 901, 4.0, 9.0, 0.0, "CA"; hetatm = true))
        s += 1; println(io, robline(s, "S", "SO4", "A", 902, 5.0, 9.0, 0.0, "S"; hetatm = true))

        s += 1; println(io, robline(s, "HA", "ALA", "A", 1, 1.0, 1.0, 1.0, "H"))
        println(io, "END")
    end

    at = read_atoms(path; T = Float64)
    @test at isa AtomTable
    @test length(at) == 10
    @test at.report.n_water == 1
    @test at.report.n_hydrogen == 1
    @test at.report.n_hetero_dropped == 2
    @test at.report.n_modified_kept == 1
    @test haskey(at.report.hetero_resnames, "CA")
    @test haskey(at.report.hetero_resnames, "SO4")
    @test "MSE" in at.resname
    @test count(==("SE"), at.element) == 1

    at2 = read_atoms(path; T = Float64, keep_hetero = true, keep_water = true)
    @test length(at2) > length(at)
    @test at2.report.n_hetero_dropped == 0

    ca = findall(i -> at2.resname[i] == "CA", eachindex(at2.coords))
    @test length(ca) == 1 && at2.element[ca[1]] == "CA"
end


@testset "Table 5 entry 10: the unstructured tail is reported by name" begin
    dir = mktempdir()
    clean = tail_pdb(joinpath(dir, "clean.pdb"); ntail = 0)
    at = read_atoms(clean; T = Float64)
    r = tail_report(at)
    @test r.natoms == 400
    @test r.rg > 0

    @test 1.0 < r.ratio < 1.6
    @test r.inflation < 2.0
    @test !warn_on_tail(at, r)
    @test_logs read_atoms(clean; T = Float64)

    prev = 0.0
    for (ntail, reach) in ((4, 60.0), (20, 60.0), (60, 80.0), (120, 100.0))
        p = tail_pdb(joinpath(dir, "tail$ntail.pdb"); ntail = ntail, reach = reach)
        a = read_atoms(p; T = Float64, check_tail = false)
        rr = tail_report(a)
        @test rr.ratio > 3.0
        @test rr.inflation > TAIL_INFLATION_WARN
        @test warn_on_tail(a, rr)

        @test occursin("GLY", rr.worst_label)

        @test 0.7 * reach < rr.r_max < reach
        prev = rr.inflation
    end
    @test prev > 0

    p = tail_pdb(joinpath(dir, "tailx.pdb"); ntail = 20, reach = 60.0)
    @test_logs (:warn,) read_atoms(p; T = Float64)
    @test_logs read_atoms(p; T = Float64, check_tail = false)

    one_atom = joinpath(dir, "one.pdb")
    open(one_atom, "w") do io
        println(io, robline(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"))
        println(io, "END")
    end
    a1 = read_atoms(one_atom; T = Float64)
    r1 = tail_report(a1)
    @test r1.natoms == 1
    @test isfinite(r1.inflation) && r1.inflation ≈ 1.0
    @test !warn_on_tail(a1, r1)
    @test_logs read_atoms(one_atom; T = Float64)

    coincident = joinpath(dir, "same.pdb")
    open(coincident, "w") do io
        for i in 1:5
            println(io, robline(i, "CA", "ALA", "A", i, 0.0, 0.0, 0.0, "C"))
        end
        println(io, "END")
    end
    a2 = read_atoms(coincident; T = Float64)
    r2 = tail_report(a2)
    @test isfinite(r2.inflation) && r2.inflation ≈ 1.0
    @test !warn_on_tail(a2, r2)

    ac = read_atoms(clean; T = Float64, center = true)
    au = read_atoms(clean; T = Float64, center = false)
    @test tail_report(ac).ratio ≈ tail_report(au).ratio
    @test tail_report(ac).rg ≈ tail_report(au).rg
end


@testset "calibration: real structures stay below the threshold" begin

    ids = filter(rob_have, ["1GGR", "1F3G", "1POH", "1FGN", "1TFH", "1AVX",
                            "1BVN", "1ACB"])
    if isempty(ids)
        @info "no structures fetched -- skipping the tail calibration testset. " *
              "bash scripts/fetch_structures.sh"
    else
        for id in ids
            at = read_atoms(joinpath(ROB_STRUCT, id * ".pdb"); T = Float64,
                            check_tail = false)
            r = tail_report(at)
            @test 1.5 < r.ratio < 2.5
            @test r.inflation < TAIL_INFLATION_WARN
            @test !warn_on_tail(at, r)
        end

        if rob_have("1AHW")
            whole = read_atoms(joinpath(ROB_STRUCT, "1AHW.pdb"); T = Float64,
                               check_tail = false)
            rw = tail_report(whole)
            @test rw.ratio > 2.0
            @test rw.inflation < TAIL_INFLATION_WARN

            half = read_atoms(joinpath(ROB_STRUCT, "1AHW.pdb"); T = Float64,
                              chains = ["A", "B", "C"], check_tail = false)
            @test tail_report(half).inflation < TAIL_INFLATION_WARN
        end
    end
end

end

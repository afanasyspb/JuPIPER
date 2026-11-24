"""
    JuPIPERBioStructuresExt

Activated automatically by `using BioStructures, JuPIPER`.

This is the whole of the file-format dependency. Everything downstream of
[`JuPIPER.AtomTable`](@ref) -- typing, radii, charges, voxelisation, the hot
loop -- is written against plain arrays and strings and never sees a
`BioStructures` type, which is the property that lets the parser sit in an
extension at all.

Structure-based benchmarks and the PDB reader tests load this extension
through the BioStructures dependency in the test environment.

!!! warning "Local names here must not shadow the accessors"
    `BioStructures` exports `coords`, `element`, `occupancy`, `chains` and
    `model` as *functions*. Naming a local array `coords` turns `coords(a)`
    into an attempt to call a `Vector`, and the error surfaces far from the
    cause. The output arrays are therefore `xyz`, `elem`, `occ` and so on.
"""
module JuPIPERBioStructuresExt

using JuPIPER
using BioStructures
using StaticArrays


const JP = JuPIPER

using JuPIPER: AtomTable, IngestReport, BioStructuresPDB, MODIFIED_RESIDUES,
               infer_element, center!


const WATER_RESNAMES = Set(uppercase.(collect(BioStructures.waterresnames)))

_wanted(::Nothing) = nothing
_wanted(c::AbstractString) = Set([String(c)])
_wanted(c) = Set(String.(c))

function JP._read_atoms(::BioStructuresPDB, path::AbstractString;
                     T::Type{<:AbstractFloat} = Float64,
                     chains = nothing,
                     model::Union{Nothing,Integer} = nothing,
                     keep_hydrogens::Bool = false,
                     keep_water::Bool = false,
                     keep_hetero::Bool = false,
                     modified_residues = MODIFIED_RESIDUES,
                     center::Bool = true)
    struc = read(path, PDBFormat)

    nums = sort(collect(keys(models(struc))))
    isempty(nums) && throw(ArgumentError("$path contains no model"))

    mnum = model === nothing ? first(nums) : Int(model)
    mnum in nums ||
        throw(ArgumentError("$path has no model $mnum (present: $(nums))"))
    mo = models(struc)[mnum]

    wanted = _wanted(chains)
    chains_present = String[]
    chains_kept = String[]

    xyz     = SVector{3,T}[]
    elem    = String[]
    aname   = String[]
    rname   = String[]
    rnum    = Int32[]
    icode   = Char[]
    cids    = String[]
    occ     = T[]
    bfac    = T[]
    isn     = Bool[]
    isc     = Bool[]

    natoms_seen = 0
    n_hydrogen = 0
    n_water = 0
    n_hetero_dropped = 0
    n_modified_kept = 0
    n_altloc = 0
    n_element_inferred = 0
    n_residues = 0
    n_cterm_complete = 0
    n_cterm_no_oxt = 0
    hetero_resnames = Dict{String,Int}()
    modified_resnames = Dict{String,Int}()

    for ch in collectchains(mo)
        cid = String(chainid(ch))
        push!(chains_present, cid)
        if wanted !== nothing && !(cid in wanted)
            continue
        end

        res_lo = Int[]
        res_hi = Int[]
        res_oxt = Bool[]

        for res in collectresidues(ch)
            resn = uppercase(strip(resname(res)))
            ats = collectatoms(res)
            natoms_seen += length(ats)

            if resn in WATER_RESNAMES
                if !keep_water
                    n_water += length(ats)
                    continue
                end
            elseif ishetero(res)
                if haskey(modified_residues, resn)
                    n_modified_kept += 1
                    modified_resnames[resn] = get(modified_resnames, resn, 0) + 1
                elseif !keep_hetero
                    n_hetero_dropped += length(ats)
                    hetero_resnames[resn] = get(hetero_resnames, resn, 0) + length(ats)
                    continue
                end
            end

            lo = length(xyz) + 1
            hasoxt = false
            for a in ats
                if !keep_hydrogens && hydrogenselector(a)
                    n_hydrogen += 1
                    continue
                end
                an = uppercase(strip(atomname(a)))
                el = uppercase(strip(element(a)))
                if isempty(el)
                    el = infer_element(resn, an)
                    n_element_inferred += 1
                end
                isdisorderedatom(a) && (n_altloc += 1)
                an == "OXT" && (hasoxt = true)

                c = coords(a)
                push!(xyz, SVector{3,T}(T(c[1]), T(c[2]), T(c[3])))
                push!(elem, el)
                push!(aname, an)
                push!(rname, resn)
                push!(rnum, Int32(resnumber(res)))
                push!(icode, inscode(res))
                push!(cids, cid)
                push!(occ, T(occupancy(a)))
                push!(bfac, T(tempfactor(a)))
                push!(isn, false)
                push!(isc, false)
            end
            hi = length(xyz)
            hi >= lo || continue
            n_residues += 1
            push!(res_lo, lo); push!(res_hi, hi); push!(res_oxt, hasoxt)
        end

        isempty(res_lo) && continue
        push!(chains_kept, cid)
        for i in res_lo[1]:res_hi[1]
            isn[i] = true
        end

        if res_oxt[end]
            for i in res_lo[end]:res_hi[end]
                isc[i] = true
            end
            n_cterm_complete += 1
        else
            n_cterm_no_oxt += 1
        end
    end

    report = IngestReport(String(path), length(nums), mnum,
                          chains_present, chains_kept,
                          natoms_seen, length(xyz),
                          n_hydrogen, n_water, n_hetero_dropped, hetero_resnames,
                          n_modified_kept, modified_resnames,
                          n_altloc, n_element_inferred, n_residues,
                          n_cterm_complete, n_cterm_no_oxt)

    at = AtomTable{T}(xyz, elem, aname, rname, rnum, icode, cids,
                      occ, bfac, BitVector(isn), BitVector(isc),
                      zero(SVector{3,T}), false, report)
    center && center!(at)
    return at
end

end

using BioStructures, FFTW, JuPIPER

# CPU example for two selected protein chains, one from each PDB.
function search_poses(rec, lig, rotations; w1=7.0, w2=1500.0, w3=2.5)
    T = eltype(rec)
    types = load_acp_types()
    pot = load_dars(; T)
    spec = gridspec_for(rec, lig)
    cpu = JuPIPER.CPUBackend()
    weights = PiperWeights{T}(w1, w2, w3)
    spectra = piper_receptor_spectra(cpu, rec, parametrize(types, rec), pot, spec;
                                     P=5, w=weights, flags=FFTW.ESTIMATE)
    ligand = piper_structure(lig, parametrize(types, lig), pot; P=5)
    pool = [workspace(cpu, spec, 7; flags=FFTW.ESTIMATE)]
    poses = Vector{JuPIPER.Pose{T}}(undef, length(rotations))
    dock!(poses, ligand, rotations.rots, spectra, pool)
    return poses, spec
end

# Keep exactly the heavy atoms used in docking. Separate the partners even
# when both input chains have the same ID, and number atoms across the complex.
function write_complex(path, rec, lig, rotation, shift)
    only(unique(rec.chain))
    only(unique(lig.chain))
    placed = pose_coords(lig.coords, rotation, shift)
    open(path, "w") do io
        serial = 0
        for (atoms, xyz, chain) in ((rec, rec.coords, "R"), (lig, placed, "L"))
            for i in eachindex(xyz)
                serial += 1
                name = atoms.atomname[i]
                if length(atoms.element[i]) == 1 && length(name) < 4 && !isdigit(first(name))
                    name = " " * name
                end
                record = BioStructures.AtomRecord(false, serial, rpad(name, 4), ' ',
                    atoms.resname[i], chain, atoms.resnum[i], atoms.inscode[i],
                    xyz[i] + rec.centroid, atoms.occupancy[i], atoms.tempfactor[i],
                    atoms.element[i], "")
                println(io, BioStructures.pdbline(record))
            end
            println(io, "TER")
        end
        println(io, "END")
    end
end

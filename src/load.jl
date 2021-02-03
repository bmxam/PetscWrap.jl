"""
    Function from GridapPETSC to find PETSc lib location.
"""
function get_petsc_location()
    PETSC_DIR   = haskey(ENV,"PETSC_DIR") ? ENV["PETSC_DIR"] : "/usr/lib/petsc"
    PETSC_ARCH  = haskey(ENV,"PETSC_ARCH") ? ENV["PETSC_ARCH"] : ""
    PETSC_LIB = ""

    # Check PETSC_DIR exists
    if isdir(PETSC_DIR)

        # Define default paths
        PETSC_LIB_DIR = joinpath(PETSC_DIR,PETSC_ARCH,"lib")

        # Check PETSC_LIB (.../libpetsc.so or .../libpetsc_real.so file) exists
        if isfile(joinpath(PETSC_LIB_DIR,"libpetsc.so"))
            PETSC_LIB = joinpath(PETSC_LIB_DIR,"libpetsc.so")
        elseif isfile(joinpath(PETSC_LIB_DIR,"libpetsc_real.so"))
            PETSC_LIB = joinpath(PETSC_LIB_DIR,"libpetsc_real.so")
        end
    end

    # PETSc lib not found
    if(length(PETSC_LIB) == 0)
        # Workaround for automerging on RegistryCI
        if(haskey(ENV,"JULIA_REGISTRYCI_AUTOMERGE"))
            PETSC_LIB = "JULIA_REGISTRYCI_AUTOMERGE"
        else
            throw(ErrorException("PETSc shared library (libpetsc.so) not found. Please check that PETSC_DIR and PETSC_ARCH env. variables are set."))
        end
    end

    return PETSC_LIB
end

# Absolute path to libpetsc.so
const libpetsc = get_petsc_location()
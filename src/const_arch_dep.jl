# Set ARCH dependant types / constants
# Directly copied or inspired from https://github.com/JuliaParallel/PETSc.jl.git

"""
    Retrieve a PETSc datatype from a String
"""
function DataTypeFromString(name::AbstractString)
    dtype_ref = Ref{PetscDataType}()
    found_ref = Ref{PetscBool}()
    ccall((:PetscDataTypeFromString, libpetsc), PetscErrorCode,
        (Cstring, Ptr{PetscDataType}, Ptr{PetscBool}),
        name, dtype_ref, found_ref)
    @assert found_ref[] == PETSC_TRUE
    return dtype_ref[]
end

"""
    Retrieve a PETSc datatype from a PETSc datatype
"""
function PetscDataTypeGetSize(dtype::PetscDataType)
    datasize_ref = Ref{Csize_t}()
    ccall((:PetscDataTypeGetSize, libpetsc), PetscErrorCode,
        (PetscDataType, Ptr{Csize_t}),
        dtype, datasize_ref)
    return datasize_ref[]
end

"""
    Find the Julia type for `PetscReal`
"""
function PetscReal2Type()
    # Workaround for RegistryCI
    (libpetsc == "JULIA_REGISTRYCI_AUTOMERGE") && (return Cdouble)

    PETSC_REAL = DataTypeFromString("Real")
    result =
        PETSC_REAL == PETSC_DOUBLE ? Cdouble :
        PETSC_REAL == PETSC_FLOAT ? Cfloat :
        error("PETSC_REAL = $PETSC_REAL not supported.")
    return result
end

"""
    Find the Julia type for PetscScalar. This function must be called after
    the line `const PetscReal = PetscReal2Type` since it uses `PetscReal`.
"""
function PetscScalar2Type()
    # Workaround for RegistryCI
    (libpetsc == "JULIA_REGISTRYCI_AUTOMERGE") && (return Cdouble)

    PETSC_REAL = DataTypeFromString("Real")
    PETSC_SCALAR = DataTypeFromString("Scalar")
    result =
        PETSC_SCALAR == PETSC_REAL ? PetscReal :
        PETSC_SCALAR == PETSC_COMPLEX ? Complex{PetscReal} :
        error("PETSC_SCALAR = $PETSC_SCALAR not supported.")
    return result
end

"""
    Find the Julia type for `PetscInt`
"""
function PetscInt2Type()
    # Workaround for RegistryCI
    (libpetsc == "JULIA_REGISTRYCI_AUTOMERGE") && (return Int32)

    PETSC_INT_SIZE = PetscDataTypeGetSize(PETSC_INT)
    result =
        PETSC_INT_SIZE == 4 ? Int32 :
        PETSC_INT_SIZE == 8 ? Int64 :
        error("PETSC_INT_SIZE = $PETSC_INT_SIZE not supported.")
    return result
end

const PetscReal = PetscReal2Type()
const PetscScalar = PetscScalar2Type()
const PetscInt = PetscInt2Type()

const PetscIntOne = PetscInt(1) # Integer `1` with the type of PetscInt, usefull to go back and forth with julia/petsc indexing
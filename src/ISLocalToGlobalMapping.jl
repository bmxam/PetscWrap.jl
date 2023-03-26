const CISLocalToGlobalMapping = Ptr{Cvoid}

struct ISLocalToGlobalMapping
    ptr::Ref{CISLocalToGlobalMapping}
    comm::MPI.Comm

    ISLocalToGlobalMapping(comm::MPI.Comm) = new(Ref{Ptr{Cvoid}}(), comm)
end

# allows us to pass ISLocalToGlobalMapping objects directly into CISLocalToGlobalMapping ccall signatures
Base.cconvert(::Type{CISLocalToGlobalMapping}, l2g::ISLocalToGlobalMapping) = l2g.ptr[]

"""
    localToGlobalMappingCreate(
        comm::MPI.Comm,
        bs::PetscInt,
        n::PetscInt,
        indices::Vector{PetscInt},
        mode::PetscCopyMode,
    )

Wrapper to `ISLocalToGlobalMappingCreate`
https://petsc.org/release/docs/manualpages/IS/ISLocalToGlobalMappingCreate/
"""
function create(
    ::Type{ISLocalToGlobalMapping},
    comm::MPI.Comm,
    bs::PetscInt,
    n::PetscInt,
    indices::Vector{PetscInt},
    mode::PetscCopyMode,
)
    l2g = ISLocalToGlobalMapping(comm)
    error = ccall(
        (:ISLocalToGlobalMappingCreate, libpetsc),
        PetscErrorCode,
        (
            MPI.MPI_Comm,
            PetscInt,
            PetscInt,
            Ptr{PetscInt},
            PetscCopyMode,
            Ptr{CISLocalToGlobalMapping},
        ),
        comm,
        bs,
        n,
        indices,
        mode,
        l2g,
    )
    @assert iszero(error)
    return l2g
end

"""
    destroy(mapping::ISLocalToGlobalMapping)

Wrapper to `ISLocalToGlobalMappingDestroy`
https://petsc.org/release/docs/manualpages/IS/ISLocalToGlobalMappingDestroy/
"""
function destroy(mapping::ISLocalToGlobalMapping)
    error = ccall(
        (:ISLocalToGlobalMappingDestroy, libpetsc),
        PetscErrorCode,
        (Ptr{CISLocalToGlobalMapping},),
        mapping.ptr,
    )
    @assert iszero(error)
end

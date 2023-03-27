const CISLocalToGlobalMapping = Ptr{Cvoid}

struct ISLocalToGlobalMapping
    ptr::Ref{CISLocalToGlobalMapping}
    comm::MPI.Comm

    ISLocalToGlobalMapping(comm::MPI.Comm) = new(Ref{Ptr{Cvoid}}(), comm)
end

# allows us to pass ISLocalToGlobalMapping objects directly into CISLocalToGlobalMapping ccall signatures
Base.cconvert(::Type{CISLocalToGlobalMapping}, l2g::ISLocalToGlobalMapping) = l2g.ptr[]

"""
    create(
        comm::MPI.Comm,
        bs::PetscInt,
        n::PetscInt,
        indices::Vector{PetscInt},
        mode::PetscCopyMode,
    )

    create(
        ::Type{ISLocalToGlobalMapping},
        comm::MPI.Comm,
        indices::Vector{I},
        mode::PetscCopyMode = PETSC_COPY_VALUES,
    ) where {I<:Integer}

Wrapper to `ISLocalToGlobalMappingCreate`
https://petsc.org/release/docs/manualpages/IS/ISLocalToGlobalMappingCreate/

0-based indexing
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
        l2g.ptr,
    )
    @assert iszero(error)
    return l2g
end

function create(
    ::Type{ISLocalToGlobalMapping},
    comm::MPI.Comm,
    indices::Vector{I},
    mode::PetscCopyMode = PETSC_COPY_VALUES,
) where {I<:Integer}
    return create(
        ISLocalToGlobalMapping,
        comm,
        PetscIntOne,
        PetscInt(length(indices)),
        PetscInt.(indices),
        mode,
    )
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

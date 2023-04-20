const CISLocalToGlobalMapping = Ptr{Cvoid}

mutable struct ISLocalToGlobalMapping
    ptr::CISLocalToGlobalMapping
    comm::MPI.Comm

    ISLocalToGlobalMapping(comm::MPI.Comm) = new(CISLocalToGlobalMapping(), comm)
end

Base.unsafe_convert(::Type{CISLocalToGlobalMapping}, x::ISLocalToGlobalMapping) = x.ptr
function Base.unsafe_convert(
    ::Type{Ptr{CISLocalToGlobalMapping}},
    x::ISLocalToGlobalMapping,
)
    Ptr{CISLocalToGlobalMapping}(pointer_from_objref(x))
end

"""
    create(
        comm::MPI.Comm,
        bs::PetscInt,
        n::PetscInt,
        indices::Vector{PetscInt},
        mode::PetscCopyMode;
        add_finalizer = true,
    )

    create(
        ::Type{ISLocalToGlobalMapping},
        comm::MPI.Comm,
        indices::Vector{I},
        mode::PetscCopyMode = PETSC_COPY_VALUES,
        mode::PetscCopyMode;
        add_finalizer = true,
    ) where {I<:Integer}

Wrapper to `ISLocalToGlobalMappingCreate`
https://petsc.org/release/manualpages/IS/ISLocalToGlobalMappingCreate/

0-based indexing
"""
function create(
    ::Type{ISLocalToGlobalMapping},
    comm::MPI.Comm,
    bs::PetscInt,
    n::PetscInt,
    indices::Vector{PetscInt},
    mode::PetscCopyMode;
    add_finalizer = true,
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

    _NREFS[] += 1
    add_finalizer && finalizer(destroy, l2g)

    return l2g
end

function create(
    ::Type{ISLocalToGlobalMapping},
    comm::MPI.Comm,
    indices::Vector{I},
    mode::PetscCopyMode = PETSC_COPY_VALUES;
    add_finalizer = true,
) where {I<:Integer}
    return create(
        ISLocalToGlobalMapping,
        comm,
        PetscIntOne,
        PetscInt(length(indices)),
        PetscInt.(indices),
        mode;
        add_finalizer,
    )
end

"""
    destroy(mapping::ISLocalToGlobalMapping)

Wrapper to `ISLocalToGlobalMappingDestroy`
https://petsc.org/release/manualpages/IS/ISLocalToGlobalMappingDestroy/
"""
function destroy(mapping::ISLocalToGlobalMapping)
    error = ccall(
        (:ISLocalToGlobalMappingDestroy, libpetsc),
        PetscErrorCode,
        (Ptr{CISLocalToGlobalMapping},),
        mapping,
    )
    @assert iszero(error)

    _NREFS[] -= 1
end

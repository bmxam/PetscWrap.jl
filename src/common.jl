abstract type AbstractPetscObject end

const CPetscObject = Ptr{Cvoid}

function objectRegisterDestroy(obj::AbstractPetscObject)
    error =
        ccall((:PetscObjectRegisterDestroy, libpetsc), PetscErrorCode, (CPetscObject,), obj)
    @assert iszero(error)
end
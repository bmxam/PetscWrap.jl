abstract type AbstractPetscObject end
_get_ptr(obj::AbstractPetscObject) = obj.ptr

const CPetscObject = Ptr{Cvoid}

function objectRegisterDestroy(obj::AbstractPetscObject)
    error =
        ccall((:PetscObjectRegisterDestroy, libpetsc), PetscErrorCode, (CPetscObject,), obj)
    @assert iszero(error)
end

_is_destroyed(obj::AbstractPetscObject) = _get_ptr(obj) == C_NULL

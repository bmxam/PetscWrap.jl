"""
    Wrapper to PetscInitializeNoArguments
"""
function PetscInitialize()
    error = ccall((:PetscInitializeNoArguments, libpetsc), PetscErrorCode, ())
    @assert iszero(error)
end

"""
    Wrapper to PetscInitializeNoPointers

# Implementation
I don't know if I am supposed to use PetscInt or not...
"""
function PetscInitialize(args::Vector{String}, filename::String, help::String)
    args2 = ["julia"; args]
    nargs = Cint(length(args2))
    error = ccall( (:PetscInitializeNoPointers, libpetsc),
            PetscErrorCode,
            (Cint,
            Ptr{Ptr{UInt8}},
            Cstring,
            Cstring),
            nargs, args2, filename, help
    )
    @assert iszero(error)
end

PetscInitialize(args::Vector{String}) = PetscInitialize(args, "", "")
PetscInitialize(args::String) = PetscInitialize(convert(Vector{String}, split(args)), "", "")

"""
    Wrapper to PetscFinalize
"""
function PetscFinalize()
    error = ccall( (:PetscFinalize, libpetsc), PetscErrorCode, ())
    @assert iszero(error)
end
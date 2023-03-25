"""
    Wrapper to `PetscInitializeNoPointers`. Initialize PETCs with arguments.

# Implementation
I don't know if I am supposed to use PetscInt or not...
"""
function PetscInitialize(args::Vector{String}, filename::String, help::String)
    args2 = ["julia"; args]
    nargs = Cint(length(args2))
    error = ccall((:PetscInitializeNoPointers, libpetsc),
        PetscErrorCode,
        (Cint,
            Ptr{Ptr{UInt8}},
            Cstring,
            Cstring),
        nargs, args2, filename, help
    )
    @assert iszero(error)
end

"""
    Initialize PETSc with arguments stored in a vector of string
"""
PetscInitialize(args::Vector{String}) = PetscInitialize(args, "", "")

"""
    Initialize PETSc with arguments concatenated in a unique string.
"""
PetscInitialize(args::String) = PetscInitialize(convert(Vector{String}, split(args)), "", "")

"""
    PetscInitialize(cmd_line_args::Bool = true)

Initialize PETSc.

If `cmd_line_args == true`, then command line arguments passed to Julia are used as
arguments for PETSc (leading to a call to `PetscInitializeNoPointers`).

Otherwise, if `cmd_line_args == false`, initialize PETSc without arguments (leading
to a call to `PetscInitializeNoArguments`).
"""
function PetscInitialize(cmd_line_args::Bool=true)
    if (cmd_line_args)
        PetscInitialize(ARGS)
    else
        error = ccall((:PetscInitializeNoArguments, libpetsc), PetscErrorCode, ())
        @assert iszero(error)
    end
end

"""
    Wrapper to PetscFinalize
"""
function PetscFinalize()
    error = ccall((:PetscFinalize, libpetsc), PetscErrorCode, ())
    @assert iszero(error)
end

"""
    Wrapper to PetscInitialized
"""
function PetscInitialized()
    isInitialized = Ref{PetscBool}()
    error = ccall((:PetscInitialized, libpetsc), PetscErrorCode, (Ref{PetscBool},), isInitialized)
    @assert iszero(error)
    return Bool(isInitialized[])
end
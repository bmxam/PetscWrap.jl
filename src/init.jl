"""
    Wrapper to `PetscInitializeNoPointers`. Initialize PETCs with arguments.

# Implementation

I don't know if I am supposed to use PetscInt or not...
"""
function PetscInitialize(args::Vector{String}, filename::String, help::String)
    args2 = ["julia"; args]
    nargs = Cint(length(args2))
    error = ccall(
        (:PetscInitializeNoPointers, libpetsc),
        PetscErrorCode,
        (Cint, Ptr{Ptr{UInt8}}, Cstring, Cstring),
        nargs,
        args2,
        filename,
        help,
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
function PetscInitialize(args::String)
    PetscInitialize(convert(Vector{String}, split(args)), "", "")
end

"""
    PetscInitialize(cmd_line_args::Bool = true)

Initialize PETSc.

If `cmd_line_args == true`, then command line arguments passed to Julia are used as
arguments for PETSc (leading to a call to `PetscInitializeNoPointers`).

Otherwise, if `cmd_line_args == false`, initialize PETSc without arguments (leading
to a call to `PetscInitializeNoArguments`).
"""
function PetscInitialize(cmd_line_args::Bool = true)
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
function PetscFinalize(finalizeMPI = false)
    GC.gc()
    if _NREFS[] != 0
        @warn "$(_NREFS[]) objects still not finalized before calling PetscWrap.Finalize()"
    end

    error = ccall((:PetscFinalize, libpetsc), PetscErrorCode, ())
    @assert iszero(error)

    finalizeMPI && MPI.Finalize()
end

"""
    Wrapper to PetscInitialized
"""
function PetscInitialized()
    isInitialized = Ref{PetscBool}()
    error = ccall(
        (:PetscInitialized, libpetsc),
        PetscErrorCode,
        (Ref{PetscBool},),
        isInitialized,
    )
    @assert iszero(error)
    return Bool(isInitialized[])
end

"""
    Wrapper to PetscFinalized

Cmd line options:
-options_view - Calls PetscOptionsView()
-options_left - Prints unused options that remain in the database
-objects_dump [all] - Prints list of objects allocated by the user that have not been freed, the option all cause all outstanding objects to be listed
-mpidump - Calls PetscMPIDump()
-malloc_dump - Calls PetscMallocDump(), displays all memory allocated that has not been freed
-malloc_info - Prints total memory usage
-malloc_view - Prints list of all memory allocated and where
"""
function PetscFinalized()
    isFinalized = Ref{PetscBool}()
    error =
        ccall((:PetscFinalized, libpetsc), PetscErrorCode, (Ref{PetscBool},), isFinalized)
    @assert iszero(error)
    return Bool(isFinalized[])
end

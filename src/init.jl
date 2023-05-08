"""
    Wrapper to `PetscInitializeNoPointers`. Initialize PETCs with arguments.

# Implementation

I don't know if I am supposed to use PetscInt or not...
"""
function PetscInitialize(
    args::Vector{String},
    filename::String,
    help::String;
    finalize_atexit = true,
)
    MPI.Initialized() || MPI.Init()

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

    finalize_atexit && atexit(PetscFinalize)
end

"""
    Initialize PETSc with arguments stored in a vector of string
"""
function PetscInitialize(args::Vector{String}; finalize_atexit = true)
    PetscInitialize(args, "", ""; finalize_atexit)
end

"""
    Initialize PETSc with arguments concatenated in a unique string.
"""
function PetscInitialize(args::String; finalize_atexit = true)
    PetscInitialize(convert(Vector{String}, split(args)), "", ""; finalize_atexit)
end

"""
    PetscInitialize(cmd_line_args::Bool = true; finalize_atexit = true)

Initialize PETSc.

If `cmd_line_args == true`, then command line arguments passed to Julia are used as
arguments for PETSc (leading to a call to `PetscInitializeNoPointers`).

Otherwise, if `cmd_line_args == false`, initialize PETSc without arguments (leading
to a call to `PetscInitializeNoArguments`).
"""
function PetscInitialize(cmd_line_args::Bool = true; finalize_atexit = true)
    if (cmd_line_args)
        PetscInitialize(ARGS; finalize_atexit)
    else
        MPI.Initialized() || MPI.Init()

        error = ccall((:PetscInitializeNoArguments, libpetsc), PetscErrorCode, ())
        @assert iszero(error)

        finalize_atexit && atexit(PetscFinalize)
    end
end

"""
    Wrapper to PetscFinalize
"""
function PetscFinalize()
    PetscFinalized() && return

    GC.gc()
    if _NREFS[] != 0
        @warn "$(_NREFS[]) objects still not finalized before calling PetscWrap.Finalize()"
    end

    error = ccall((:PetscFinalize, libpetsc), PetscErrorCode, ())
    @assert iszero(error)

    if _NREFS[] != 0
        @warn "$(_NREFS[]) objects still not finalized after calling PetscWrap.Finalize()"
    end

    _NREFS[] = 0
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

get_PETSC_COMM_WORLD() = cglobal((:PETSC_COMM_WORLD, libpetsc), PetscInt)

const CViewer = Ptr{Cvoid}
struct PetscViewer
    ptr::Ref{CViewer}

    PetscViewer() = new(Ref{CViewer}())
end

# allows us to pass PetscViewer objects directly into CViewer ccall signatures
Base.cconvert(::Type{CViewer}, viewer::PetscViewer) = viewer.ptr[]

"""
    PetscViewerASCIIGetStdout(comm::MPI.Comm, viewer::PetscViewer)

Wrapper for `PetscViewerASCIIGetStdout`
"""
function PetscViewerASCIIGetStdout(comm::MPI.Comm, viewer::PetscViewer)
    error = ccall((:PetscViewerASCIIGetStdout, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CViewer}), comm, viewer.ptr)
    @assert iszero(error)
end

function PetscViewerASCIIGetStdout(comm::MPI.Comm)
    viewer = PetscViewer()
    error = ccall((:PetscViewerASCIIGetStdout, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CViewer}), comm, viewer.ptr)
    @assert iszero(error)
    return viewer
end

PetscViewerStdWorld() = PetscViewerASCIIGetStdout(MPI.COMM_WORLD)

"""
    PetscViewerCreate(comm::MPI.Comm, viewer::PetscViewer)

Wrapper for `PetscViewerCreate`
"""
function PetscViewerCreate(comm::MPI.Comm, viewer::PetscViewer)
    error = ccall((:PetscViewerCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CViewer}), comm, viewer.ptr)
    @assert iszero(error)
end

"""
    PetscViewerPushFormat(viewer::PetscViewer, format::PetscViewerFormat)

Wrapper for `PetscViewerPushFormat`
"""
function PetscViewerPushFormat(viewer::PetscViewer, format::PetscViewerFormat)
    error = ccall((:PetscViewerPushFormat, libpetsc), PetscErrorCode, (CViewer, PetscViewerFormat), viewer, format)
    @assert iszero(error)
end

"""
    PetscViewerPopFormat(viewer::PetscViewer)

Wrapper for `PetscViewerPopFormat`
"""
function PetscViewerPopFormat(viewer::PetscViewer)
    error = ccall((:PetscViewerPopFormat, libpetsc), PetscErrorCode, (CViewer,), viewer)
    @assert iszero(error)
end

"""
    PetscViewerASCIIOpen(comm::MPI.Comm, filanem, viewer::PetscViewer)

Wrapper for `PetscViewerASCIIOpen`
"""
function PetscViewerASCIIOpen(comm::MPI.Comm, filename, viewer::PetscViewer)
    error = ccall((:PetscViewerASCIIOpen, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Cstring, Ptr{CViewer}), comm, filename, viewer.ptr)
    @assert iszero(error)
end

"""
    PetscViewerDestroy(viewer::PetscViewer)

Wrapper for `PetscViewerDestroy`

Warning : from what I understand, all viewers must not be destroyed explicitely using `PetscViewerDestroy`.

"""
function PetscViewerDestroy(viewer::PetscViewer)
    error = ccall((:PetscViewerDestroy, libpetsc), PetscErrorCode, (Ptr{CViewer}, ), viewer.ptr)
    @assert iszero(error)
end
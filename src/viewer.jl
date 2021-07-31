const CViewer = Ptr{Cvoid}
struct PetscViewer
    ptr::Ref{CViewer}
    comm::MPI.Comm

    PetscViewer(comm::MPI.Comm) = new(Ref{CViewer}(), comm)
end

# allows us to pass PetscViewer objects directly into CViewer ccall signatures
Base.cconvert(::Type{CViewer}, viewer::PetscViewer) = viewer.ptr[]

"""
    PetscViewerASCIIGetStdout(comm::MPI.Comm = MPI.COMM_WORLD)

Wrapper for `PetscViewerASCIIGetStdout`
"""
function PetscViewerASCIIGetStdout(comm::MPI.Comm = MPI.COMM_WORLD)
    viewer = PetscViewer(comm)
    error = ccall((:PetscViewerASCIIGetStdout, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CViewer}), comm, viewer.ptr)
    @assert iszero(error)
    return viewer
end

const PetscViewerStdWorld = PetscViewerASCIIGetStdout

"""
    PetscViewerCreate(comm::MPI.Comm = MPI.COMM_WORLD)

Wrapper for `PetscViewerCreate`
"""
function PetscViewerCreate(comm::MPI.Comm = MPI.COMM_WORLD)
    viewer = PetscViewer(comm)
    error = ccall((:PetscViewerCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CViewer}), comm, viewer.ptr)
    @assert iszero(error)
    return viewer
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
    PetscViewerASCIIOpen(comm::MPI.Comm, filename)

Wrapper for `PetscViewerASCIIOpen`
"""
function PetscViewerASCIIOpen(comm::MPI.Comm, filename)
    viewer = PetscViewer(comm)
    error = ccall((:PetscViewerASCIIOpen, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Cstring, Ptr{CViewer}), comm, filename, viewer.ptr)
    @assert iszero(error)
    return viewer
end

"""
PetscViewerFileSetName(viewer::PetscViewer, filename)

Wrapper for `PetscViewerFileSetName`
"""
function PetscViewerFileSetName(viewer::PetscViewer, filename::String)
    error = ccall((:PetscViewerFileSetName, libpetsc), PetscErrorCode, (CViewer, Cstring), viewer, filename)
    @assert iszero(error)
end

"""
PetscViewerFileSetMode(viewer::PetscViewer, mode::PetscFileMode = FILE_MODE_WRITE)

Wrapper for `PetscViewerFileSetMode`
"""
function PetscViewerFileSetMode(viewer::PetscViewer, mode::PetscFileMode = FILE_MODE_WRITE)
    error = ccall((:PetscViewerFileSetMode, libpetsc), PetscErrorCode, (CViewer, PetscFileMode), viewer, mode)
    @assert iszero(error)
end

"""
    PetscViewerHDF5Open(comm::MPI.Comm, filename::String, type::PetscFileMode)

Wrapper for `PetscViewerHDF5Open`
"""
function PetscViewerHDF5Open(comm::MPI.Comm, filename::String, type::PetscFileMode)
    viewer = PetscViewer(comm)
    error = ccall((:PetscViewerHDF5Open, libpetsc), PetscErrorCode,
        (MPI.MPI_Comm, Cstring, PetscFileMode, Ptr{CViewer}),
        comm, filename, type, viewer.ptr)
    @assert iszero(error)
    return viewer
end

"""
    PetscViewerSetType(viewer::PetscViewer, type::String)

Wrapper for `PetscViewerSetType`. Values for `type` alors available here:
https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PetscViewerType.html#PetscViewerType
"""
function PetscViewerSetType(viewer::PetscViewer, type::String)
    error = ccall((:PetscViewerSetType, libpetsc), PetscErrorCode, (CViewer, Cstring), viewer, type)
    @assert iszero(error)
end

"""
    PetscViewerView(v::PetscViewer, viewer::PetscViewer = PetscViewerStdWorld())

Wrapper to `PetscViewerView`
"""
function PetscViewerView(v::PetscViewer, viewer::PetscViewer = PetscViewerStdWorld())
    error = ccall((:PetscViewerView, libpetsc), PetscErrorCode, (CViewer, CViewer), v, viewer)
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
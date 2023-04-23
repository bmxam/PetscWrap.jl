# DEV NOTE : using `finalizer` for PetscViewer failed because for some reason I don't understand,
# calling `PetscViewerDestroy` often fails (even outside the context of `finalizer`)

const CViewer = Ptr{Cvoid}
mutable struct PetscViewer <: AbstractPetscObject
    ptr::CViewer
    comm::MPI.Comm

    PetscViewer(comm::MPI.Comm) = new(CViewer(), comm)
end

Base.unsafe_convert(::Type{CViewer}, x::PetscViewer) = x.ptr
function Base.unsafe_convert(::Type{Ptr{CViewer}}, x::PetscViewer)
    Ptr{CViewer}(pointer_from_objref(x))
end

"""
    ASCIIGetStdout(comm::MPI.Comm)

Wrapper for `PetscViewerASCIIGetStdout`
https://petsc.org/release/manualpages/Viewer/PetscViewerASCIIGetStdout/
"""
function ASCIIGetStdout(comm::MPI.Comm = MPI.COMM_WORLD)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerASCIIGetStdout, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CViewer}),
        comm,
        viewer,
    )
    @assert iszero(error)
    return viewer
end

const StdWorld = ASCIIGetStdout

"""
    ASCIIOpen(comm::MPI.Comm, name)

Wrapper for `PetscViewerASCIIOpen`
https://petsc.org/release/manualpages/Viewer/PetscViewerASCIIOpen/
"""
function ASCIIOpen(comm::MPI.Comm, name::String)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerASCIIOpen, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Cstring, Ptr{CViewer}),
        comm,
        name,
        viewer,
    )
    @assert iszero(error)
    return viewer
end

"""
    create(::Type{PetscViewer}, comm::MPI.Comm = MPI.COMM_WORLD)

Wrapper for `PetscViewerCreate`
https://petsc.org/release/manualpages/Viewer/PetscViewerCreate/
"""
function create(::Type{PetscViewer}, comm::MPI.Comm = MPI.COMM_WORLD)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerCreate, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CViewer}),
        comm,
        viewer,
    )
    @assert iszero(error)
    return viewer
end

"""
    destroy(viewer::PetscViewer)

Wrapper for `PetscViewerDestroy`
https://petsc.org/release/manualpages/Viewer/PetscViewerDestroy/

Warning : from what I understand, all viewers must not be destroyed explicitely using `PetscViewerDestroy`.
"""
function destroy(viewer::PetscViewer)
    _is_destroyed(viewer) && return

    error = ccall((:PetscViewerDestroy, libpetsc), PetscErrorCode, (Ptr{CViewer},), viewer)
    @assert iszero(error)
end

"""
    fileSetMode(viewer::PetscViewer, mode::PetscFileMode)

Wrapper for `PetscViewerFileSetMode`
https://petsc.org/release/manualpages/Viewer/PetscViewerFileSetMode/
"""
function fileSetMode(viewer::PetscViewer, mode::PetscFileMode)
    error = ccall(
        (:PetscViewerFileSetMode, libpetsc),
        PetscErrorCode,
        (CViewer, PetscFileMode),
        viewer,
        mode,
    )
    @assert iszero(error)
end

"""
    fileSetName(viewer::PetscViewer, name::String)

Wrapper for `PetscViewerFileSetName`
https://petsc.org/release/manualpages/Viewer/PetscViewerFileSetName/
"""
function fileSetName(viewer::PetscViewer, name::String)
    error = ccall(
        (:PetscViewerFileSetName, libpetsc),
        PetscErrorCode,
        (CViewer, Cstring),
        viewer,
        name,
    )
    @assert iszero(error)
end

"""
    HDF5Open(comm::MPI.Comm, name::String, type::PetscFileMode)

Wrapper for `PetscViewerHDF5Open`
https://petsc.org/release/manualpages/Viewer/PetscViewerHDF5Open/
"""
function HDF5Open(comm::MPI.Comm, name::String, type::PetscFileMode)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerHDF5Open, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Cstring, PetscFileMode, Ptr{CViewer}),
        comm,
        name,
        type,
        viewer,
    )
    @assert iszero(error)
    return viewer
end

"""
    popFormat(viewer::PetscViewer)

Wrapper for `PetscViewerPopFormat`
https://petsc.org/release/manualpages/Viewer/PetscViewerPopFormat/
"""
function popFormat(viewer::PetscViewer)
    error = ccall((:PetscViewerPopFormat, libpetsc), PetscErrorCode, (CViewer,), viewer)
    @assert iszero(error)
end

"""
    pushFormat(viewer::PetscViewer, format::PetscViewerFormat)

Wrapper for `PetscViewerPushFormat`
https://petsc.org/release/manualpages/Viewer/PetscViewerPushFormat/
"""
function pushFormat(viewer::PetscViewer, format::PetscViewerFormat)
    error = ccall(
        (:PetscViewerPushFormat, libpetsc),
        PetscErrorCode,
        (CViewer, PetscViewerFormat),
        viewer,
        format,
    )
    @assert iszero(error)
end

"""
    setType(viewer::PetscViewer, type::String)

Wrapper for `PetscViewerSetType`
https://petsc.org/release/manualpages/Viewer/PetscViewerSetType/

Values for `type` alors available here:
https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PetscViewerType.html#PetscViewerType
"""
function setType(viewer::PetscViewer, type::String)
    error = ccall(
        (:PetscViewerSetType, libpetsc),
        PetscErrorCode,
        (CViewer, Cstring),
        viewer,
        type,
    )
    @assert iszero(error)
end

"""
    viewerView(v::PetscViewer, viewer::PetscViewer)

Wrapper to `PetscViewerView`
"""
function viewerView(v::PetscViewer, viewer::PetscViewer)
    error =
        ccall((:PetscViewerView, libpetsc), PetscErrorCode, (CViewer, CViewer), v, viewer)
    @assert iszero(error)
end

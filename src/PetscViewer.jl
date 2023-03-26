const CViewer = Ptr{Cvoid}
struct PetscViewer
    ptr::Ref{CViewer}
    comm::MPI.Comm

    PetscViewer(comm::MPI.Comm) = new(Ref{CViewer}(), comm)
end

# allows us to pass PetscViewer objects directly into CViewer ccall signatures
Base.cconvert(::Type{CViewer}, viewer::PetscViewer) = viewer.ptr[]

"""
    ASCIIGetStdout(comm::MPI.Comm)

Wrapper for `PetscViewerASCIIGetStdout`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerASCIIGetStdout/
"""
function ASCIIGetStdout(comm::MPI.Comm)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerASCIIGetStdout, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CViewer}),
        comm,
        viewer.ptr,
    )
    @assert iszero(error)
    return viewer
end

const StdWorld = ASCIIGetStdout

"""
    ASCIIOpen(comm::MPI.Comm, name)

Wrapper for `PetscViewerASCIIOpen`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerASCIIOpen/
"""
function ASCIIOpen(comm::MPI.Comm, name::String)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerASCIIOpen, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Cstring, Ptr{CViewer}),
        comm,
        name,
        viewer.ptr,
    )
    @assert iszero(error)
    return viewer
end

"""
    create(::Type{PetscView}, comm::MPI.Comm)

Wrapper for `PetscViewerCreate`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerCreate/
"""
function create(::Type{PetscView}, comm::MPI.Comm)
    viewer = PetscViewer(comm)
    error = ccall(
        (:PetscViewerCreate, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CViewer}),
        comm,
        viewer.ptr,
    )
    @assert iszero(error)
    return viewer
end

"""
    destroy(viewer::PetscViewer)

Wrapper for `PetscViewerDestroy`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerDestroy/

Warning : from what I understand, all viewers must not be destroyed explicitely using `PetscViewerDestroy`.
"""
function destroy(viewer::PetscViewer)
    error =
        ccall((:PetscViewerDestroy, libpetsc), PetscErrorCode, (Ptr{CViewer},), viewer.ptr)
    @assert iszero(error)
end

"""
    popFormat(viewer::PetscViewer)

Wrapper for `PetscViewerPopFormat`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerPopFormat/
"""
function popFormat(viewer::PetscViewer)
    error = ccall((:PetscViewerPopFormat, libpetsc), PetscErrorCode, (CViewer,), viewer)
    @assert iszero(error)
end

"""
    pushFormat(viewer::PetscViewer, format::PetscViewerFormat)

Wrapper for `PetscViewerPushFormat`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerPushFormat/
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
    fileSetMode(viewer::PetscViewer, mode::PetscFileMode)

Wrapper for `PetscViewerFileSetMode`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerFileSetMode/
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
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerFileSetName/
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
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerHDF5Open/
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
        viewer.ptr,
    )
    @assert iszero(error)
    return viewer
end

"""
    setType(viewer::PetscViewer, type::String)

Wrapper for `PetscViewerSetType`
https://petsc.org/release/docs/manualpages/Viewer/PetscViewerSetType/

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
    view(v::PetscViewer, viewer::PetscViewer)

Wrapper to `PetscViewerView`
"""
function view(v::PetscViewer, viewer::PetscViewer)
    error =
        ccall((:PetscViewerView, libpetsc), PetscErrorCode, (CViewer, CViewer), v, viewer)
    @assert iszero(error)
end
const push_format! = PetscViewerPushFormat
const set_mode! = PetscViewerFileSetMode
const set_name! = PetscViewerFileSetName
const set_type! = PetscViewerSetType

"""
    PetscViewer(comm::MPI.Comm, filename::String, format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV, type::String = "ascii", mode::PetscFileMode = FILE_MODE_WRITE)

Constructor for a PetscViewer intended to read/write a matrix or a vector with the supplied type and format.
"""
function PetscViewer(
    comm::MPI.Comm,
    filename::String,
    format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
    type::String = "ascii",
    mode::PetscFileMode = FILE_MODE_WRITE,
)
    viewer = PetscViewerCreate(comm)
    set_type!(viewer, type)
    set_mode!(viewer, FILE_MODE_WRITE)
    push_format!(viewer, format)
    set_name!(viewer, filename)
    return viewer
end

destroy!(viewer::PetscViewer) = PetscViewerDestroy(viewer)

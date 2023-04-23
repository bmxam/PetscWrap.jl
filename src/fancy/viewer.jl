const push_format! = pushFormat
const set_mode! = fileSetMode
const set_name! = fileSetName
const set_type! = setType

"""
    PetscViewer(
        comm::MPI.Comm,
        filename::String,
        format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
        type::String = "ascii",
        mode::PetscFileMode = FILE_MODE_WRITE,
    )

Constructor for a PetscViewer intended to read/write a matrix or a vector with the supplied type and format.
"""
function PetscViewer(
    comm::MPI.Comm,
    filename::String,
    format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
    type::String = "ascii",
    mode::PetscFileMode = FILE_MODE_WRITE,
)
    viewer = create(PetscViewer, comm)
    set_type!(viewer, type)
    set_mode!(viewer, mode)
    push_format!(viewer, format)
    set_name!(viewer, filename)
    return viewer
end

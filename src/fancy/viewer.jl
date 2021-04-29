const push_format! = PetscViewerPushFormat
const set_mode! = PetscViewerFileSetMode
const set_name! = PetscViewerFileSetName
const set_type! = PetscViewerSetType

destroy!(viewer::PetscViewer) = PetscViewerDestroy(viewer)
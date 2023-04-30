# Fake dependency file for JULIA_REGISTRYCI_AUTOMERGE and DOC_DEPLOYMENT
const libpetsc_found = false
const libpetsc_provider = "FAKE_PETSC"
const libpetsc = "fake-path"
const PetscReal = Float64
const PetscScalar = Float64
const PetscInt = Int32
const PetscIntOne = PetscInt(1)
function show_petsc_path() end
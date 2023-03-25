# https://petsc.org/release/src/vec/vec/tutorials/ex9f.F90.html
module ghosts
using PetscWrap
using MPI

const nlocal = PetscInt(6)
const nghost = PetscInt(2)

PetscInitialize()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
mySize = MPI.Comm_size(comm)

@assert mySize == 2 "Requires 2 processors"

ifrom = (rank == 0) ? PetscInt[11, 6] : PetscInt[0, 5]

gxs = VecCreateGhost(comm, nlocal, PETSC_DECIDE, nghost, ifrom)

gx = VecDuplicate(gxs)
VecDestroy(gxs)

lx = VecGhostGetLocalForm(gx)
rstart, rend = VecGetOwnershipRange(gx)

ione = PetscInt(1)
for i in rstart:rend-1
    VecSetValues(gx, ione, PetscInt[i], PetscScalar[i])
end

VecAssemblyBegin(gx)
VecAssemblyEnd(gx)

VecGhostUpdateBegin(gx, INSERT_VALUES, SCATTER_FORWARD)
VecGhostUpdateEnd(gx, INSERT_VALUES, SCATTER_FORWARD)

# view
# VecView(lx) # wrong
array, array_ref = VecGetArray(lx)
PetscWrap.@one_at_a_time display(array)

VecGhostRestoreLocalForm(gx, lx)
VecDestroy(gx)

PetscFinalize()

end
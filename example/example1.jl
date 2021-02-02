module Example1 #hide
# # A first demo
# This example serves as a test since this project doesn't have a "testing" procedure yet. In this example,
# the equation ``u'(x) = 2`` with ``u(0) = 0`` is solved on the domain ``[0,1]`` using a backward finite
# difference scheme.
#
# In this example, PETSc legacy method names are used. For more fancy names, check the next example.
#
# Note that the way we achieve things in the document can be highly improved and the purpose of this example
# is only demonstrate some method calls to give an overview.
#
# To run this example, execute : `mpirun -n your_favorite_positive_integer julia example1.jl`

# Import package
using JuliaPetsc

# Initialize PETSc. Either without arguments, calling `PetscInitialize()` or using "command-line" arguments.
# To do so, either provide the arguments as one string, for instance
# `PetscInitialize("-ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always")` or provide each argument in
# separate strings : `PetscInitialize(["-ksp_monitor_short", "-ksp_gmres_cgs_refinement_type", "refine_always")`
PetscInitialize()

# Number of mesh points and mesh step
n = 11
Δx = 1. / (n - 1)

# Create a matrix and a vector
A = MatCreate()
b = VecCreate()

# Set the size of the different objects, leaving PETSC to decide how to distribute. Note that we should
# set the number of preallocated non-zeros to increase performance.
MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
VecSetSizes(b, PETSC_DECIDE, n)

# We can then use command-line options to set our matrix/vectors.
MatSetFromOptions(A)
VecSetFromOptions(b)

# Finish the set up
MatSetUp(A)
VecSetUp(b)

# Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
# The `rstart, rend = *GetOwnershipRange` methods differ a little bit from PETSc API since the two integers it
# returns are the effective Julia range of rows handled by the local processor. If `n` is the total
# number of rows, then `rstart ∈ [1,n]` and `rend` is the last row handled by the local processor.
b_start, b_end = VecGetOwnershipRange(b)

# Now let's build the right hand side vector. Their are various ways to do this, this is just one.
n_loc = VecGetLocalSize(b) ## Note that n_loc = b_end - b_start + 1...
VecSetValues(b, collect(b_start:b_end), 2 * ones(n_loc))

# And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.
A_start, A_end = MatGetOwnershipRange(A)
for i in A_start:A_end
    A[i, i-1:i] = [-1. 1.] / Δx
end

# Set boundary condition (only the proc handling index `1` is acting)
(b_start == 1) && (b[1] = 0.)

# Assemble matrices
MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY)
VecAssemblyBegin(b)
MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY)
VecAssemblyEnd(b)

# At this point, you can inspect `A` and `b` using the viewers (only stdout for now), simply call
# `MatView(A)` and/or `VecView(b)`

# Set up the linear solver
ksp = KSPCreate()
KSPSetOperators(ksp, A, A)
KSPSetFromOptions(ksp)
KSPSetUp(ksp)

# Solve the system. We first allocate the solution using the `VecDuplicate` method.
x = VecDuplicate(b)
KSPSolve(ksp, b, x)

# Print the solution
VecView(x)

# Free memory
MatDestroy(A)
VecDestroy(b)
VecDestroy(x)

# Finalize Petsc
PetscFinalize()

end #hide
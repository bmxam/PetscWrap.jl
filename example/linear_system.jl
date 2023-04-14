module Example #hide
# # A first demo
# This example serves as a test since this project doesn't have a "testing" procedure yet. In this example,
# the equation ``u'(x) = 2`` with ``u(0) = 0`` is solved on the domain ``[0,1]`` using a backward finite
# difference scheme.
#
# In this example, PETSc classic method names are used. For more fancy names, check the fancy version.
#
# Note that the way we achieve things in the document can be highly improved and the purpose of this example
# is only demonstrate some method calls to give an overview.
#
# To run this example, execute : `mpirun -n your_favorite_positive_integer julia example1.jl`

# Import package
using PetscWrap

# Initialize PETSc. Command line arguments passed to Julia are parsed by PETSc. Alternatively, you can
# also provide "command line arguments by defining them in a string, for instance
# `PetscInitialize("-ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always")` or by providing each argument in
# separate strings : `PetscInitialize(["-ksp_monitor_short", "-ksp_gmres_cgs_refinement_type", "refine_always")`
PetscInitialize()

# Number of mesh points and mesh step
n = 11
Δx = 1.0 / (n - 1)

# Create a matrix and a vector (you can specify the MPI communicator if you want)
A = create(Mat)
b = create(Vec)

# Set the size of the different objects, leaving PETSC to decide how to distribute. Note that we should
# set the number of preallocated non-zeros to increase performance.
setSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
setSizes(b, PETSC_DECIDE, n)

# We can then use command-line options to set our matrix/vectors.
setFromOptions(A)
setFromOptions(b)

# Finish the set up
setUp(A)
setUp(b)

# Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
# As in PETSc, the `rstart, rend = *GetOwnershipRange` methods indicate the first row handled by the local processor
# (starting at 0), and the last row (which is `rend-1`). This may be disturbing for non-regular PETSc users. Checkout
# the fancy version of this example for a more Julia-like convention.
b_start, b_end = getOwnershipRange(b)

# Now let's build the right hand side vector. Their are various ways to do this, this is just one.
n_loc = getLocalSize(b) # Note that n_loc = b_end - b_start...
setValues(b, collect(b_start:(b_end - 1)), 2 * ones(n_loc))

# And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.
A_start, A_end = getOwnershipRange(A)
for i = A_start:(A_end - 1)
    setValues(A, [i], [i - 1, i], [-1.0 1.0] / Δx, INSERT_VALUES) # setValues(A, I, J, V, INSERT_VALUES)
end

# Set boundary condition (only the proc handling index `0` is acting)
(b_start == 0) && setValue(b, 0, 0.0)

# Assemble matrices
assemblyBegin(A, MAT_FINAL_ASSEMBLY)
assemblyBegin(b)
assemblyEnd(A, MAT_FINAL_ASSEMBLY)
assemblyEnd(b)

# At this point, you can inspect `A` or `b` using a viewer (stdout by default), simply call
PetscWrap.view(A)
PetscWrap.view(b)

# Set up the linear solver
ksp = create(KSP)
setOperators(ksp, A, A)
setFromOptions(ksp)
setUp(ksp)

# Solve the system. We first allocate the solution using the `VecDuplicate` method.
x = duplicate(b)
solve(ksp, b, x)

# Print the solution
PetscWrap.view(x)

# Access the solution (this part is under development), getting a Julia array; and then restore it
array, ref = getArray(x) # do something with array
@show array
restoreArray(x, ref)

# Free memory
destroy(A)
destroy(b)
destroy(x)

# Finalize Petsc
PetscFinalize(true)

end #hide
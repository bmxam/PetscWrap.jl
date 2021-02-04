@testset "test1" begin
# Only on one processor...

# Initialize PETSc
PetscInitialize()

# Number of mesh points and mesh step
n = 11
Δx = 1. / (n - 1)

# Create a matrix and a vector
A = MatCreate()
b = VecCreate()

# Set the size of the different objects
MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
VecSetSizes(b, PETSC_DECIDE, n)

# We can then use command-line options to set our matrix/vectors.
MatSetFromOptions(A)
VecSetFromOptions(b)

# Finish the set up
MatSetUp(A)
VecSetUp(b)

# RHS range
b_start, b_end = VecGetOwnershipRange(b)

# Now let's build the right hand side vector. Their are various ways to do this, this is just one.
n_loc = VecGetLocalSize(b)
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

# Set up the linear solver
ksp = KSPCreate()
KSPSetOperators(ksp, A, A)
KSPSetFromOptions(ksp)
KSPSetUp(ksp)

# Solve the system. We first allocate the solution using the `VecDuplicate` method.
x = VecDuplicate(b)
KSPSolve(ksp, b, x)

# Print the solution
#VecView(x)

# Access the solution (this part is under development), getting a Julia array; and then restore it
array, ref = VecGetArray(x) # do something with array
@test isapprox(array, range(0., 2.; length = n))
VecRestoreArray(x, ref)

# Free memory
MatDestroy(A)
VecDestroy(b)
VecDestroy(x)
PetscFinalize()

# Reach this point?
@test true

end

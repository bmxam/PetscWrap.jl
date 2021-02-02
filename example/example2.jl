module Example1 #hide
# # Same with fancy names
# This example does more or less the same things as the first example, but using non regular API function names.
#
# This example serves as a test since this project doesn't have a "testing" procedure yet. In this example,
# the equation ``u'(x) = 2`` with ``u(0) = 0`` is solved on the domain ``[0,1]`` using a backward finite
# difference scheme.
#
# Note that the way we achieve things in the document can be highly improved and the purpose of this example
# is only demonstrate some method calls to give an overview.
#
# To run this example, execute : `mpirun -n your_favorite_positive_integer julia example2.jl`

# Import package
using PetscWrap

# Initialize PETSc. Either without arguments, calling `PetscInitialize()` or using "command-line" arguments.
# To do so, either provide the arguments as one string, for instance
# `PetscInitialize("-ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always")` or provide each argument in
# separate strings : `PetscInitialize(["-ksp_monitor_short", "-ksp_gmres_cgs_refinement_type", "refine_always")`
PetscInitialize()

# Number of mesh points and mesh step
n = 11
Δx = 1. / (n - 1)

# Create a matrix of size `(n,n)` and a vector of size `(n)`
A = create_matrix(n, n)
b = create_vector(n)

# We can then use command-line options to set our matrix/vectors.
set_from_options!(A)
set_from_options!(b)

# Finish the set up
set_up!(A)
set_up!(b)

# Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
# The `rstart, rend = *GetOwnershipRange` methods differ a little bit from PETSc API since the two integers it
# returns are the effective Julia range of rows handled by the local processor. If `n` is the total
# number of rows, then `rstart ∈ [1,n]` and `rend` is the last row handled by the local processor.
b_start, b_end = get_range(b)

# Now let's build the right hand side vector. Their are various ways to do this, this is just one.
n_loc = length(b) ## Note that n_loc = b_end - b_start + 1...
b[b_start:b_end] = 2 * ones(n_loc)

# And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.
A_start, A_end = get_range(A)
for i in A_start:A_end
    A[i, i-1:i] = [-1. 1.] / Δx
end

# Set boundary condition (only the proc handling index `1` is acting)
(b_start == 1) && (b[1] = 0.)

# Assemble matrices
assemble!(A)
assemble!(b)

# At this point, you can inspect `A` and `b` using the viewers (only stdout for now), simply call
# `MatView(A)` and/or `VecView(b)`

# Set up the linear solver
ksp = create_ksp(A)
set_from_options!(ksp)
set_up!(ksp)

# Solve the system
x = solve(ksp, b)

# Print the solution
VecView(x)

# Free memory
destroy!(A)
destroy!(b)
destroy!(x)

# Finalize Petsc
PetscFinalize()

end #hide
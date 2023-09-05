module Example #hide
using MPI
using MPIUtils
using PetscWrap
using LinearMaps
using HauntedArrays
using HauntedArrays2PetscWrap

const disable_gc = false

disable_gc && GC.enable(false)

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

const lx = 1.0
const nx = 10 ÷ np # on each process

mypart = rank + 1
Δx = lx / (np * nx - 1)

# The "mesh" partitioning
lid2gid, lid2part = HauntedArrays.generate_1d_partitioning(nx, mypart, np)

x_h = HauntedVector(comm, lid2gid, lid2part; cacheType = PetscCache)
y_h = similar(x_h)

f!(y::HauntedVector, x::HauntedVector) =
    for li in eachindex(x)
        gi = local_to_global(x, li)
        if gi == 1
            y[li] = x[li]
        else
            for lj in eachindex(x)
                gj = local_to_global(x, lj)
                if gj == gi - 1
                    y[li] = (x[li] - x[lj]) / Δx
                end
            end
        end
    end

function f!(y, x)
    println("calling custom mult operator...")

    # Update HauntedVectors values `x_h` with Petsc values `x`
    println("updating HauntedVector...")
    HauntedArrays2PetscWrap.update!(x_h, x)
    HauntedArrays.update_ghosts!(x_h)

    # Apply the operator to obtain a result as a HauntedVector (`y_h`)
    println("Applying operator...")
    f!(y_h, x_h)

    # Update Petsc values `y` with the HauntedVector values (`y_h`)
    println("Updating Petsc vector...")
    HauntedArrays2PetscWrap.update!(y, y_h, get_cache(y_h).oid2pid0)

    # MPI.Barrier(PetscWrap._get_comm(y))
    println("done ! ")
end

# Create a matrix and a vector
b = create(Vec, comm)

A = PetscWrap.createShell(comm, Int32(nx), Int32(nx))
cfunc = PetscWrap.set_shell_mul!(A, f!)

# function _f!(::CMat, x::CVec, y::CVec)::Cint
#     comm = PetscWrap._get_comm(A)
#     f!(Vec(comm, y), Vec(comm, x))
#     return PetscErrorCode(0) # return "success" (mandatory)
# end
# shell_mul_c = @cfunction($_f!, Cint, (CMat, CVec, CVec))
# PetscWrap.shellSetOperation(A, PetscWrap.MATOP_MULT, shell_mul_c)

setSizes(b, PETSC_DECIDE, nx * np)

# We can then use command-line options to set our matrix/vectors.
setFromOptions(A)
setFromOptions(b)

# Finish the set up
setUp(A)
setUp(b)

b_start, b_end = getOwnershipRange(b)
n_loc = getLocalSize(b) # Note that n_loc = b_end - b_start...
setValues(b, collect(b_start:(b_end - 1)), 2 * ones(n_loc))

# Set boundary condition (only the proc handling index `0` is acting)
(b_start == 0) && setValue(b, 0, 0.0)

# Assemble matrices
assemblyBegin(A, MAT_FINAL_ASSEMBLY)
assemblyBegin(b)
assemblyEnd(A, MAT_FINAL_ASSEMBLY)
assemblyEnd(b)

# At this point, you can inspect `A` or `b` using a viewer (stdout by default), simply call
# matView(A)
# vecView(b)

# Set up the linear solver
println("setting up")
ksp = create(KSP, comm)
setOperators(ksp, A, A)
setFromOptions(ksp)
setUp(ksp)

# Solve the system. We first allocate the solution using the `VecDuplicate` method.
x = duplicate(b)
println("solving")
solve(ksp, b, x)

# Print the solution
println("viewing")
vecView(x)

@only_root @show [2 * (i - 1) * Δx for i = 1:(np * nx)]

# Free memory. Note that this call is faculative since, by default,
# the julia GC will trigger a call to Petsc `destroy` to each object
destroy.((ksp, A, b, x))

# x_h .= 0.0
# y_h .= 0.0
disable_gc && GC.enable(true)

end #hide

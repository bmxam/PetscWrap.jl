# PetscWrap.jl

PetscWrap.jl is a parallel Julia wrapper for the (awesome) [PETSc](https://www.mcs.anl.gov/petsc/) library. It can be considered as a fork from the [GridapPetsc.jl](https://github.com/gridap/GridapPETSc.jl) and [Petsc.jl](https://github.com/JuliaParallel/PETSc.jl) projects.

The main differences with the two aformentionned projects are:
- parallel support : you can solve linear systems on multiple core with `mpirun -n 4 julia foo.jl`;
- no dependance to other Julia packages except `MPI.jl`;
- possibility to switch from one PETSc "arch" to another without recompiling the project;
- less PETSc API functions wrappers for now.

Note that the primary objective of this project is to enable the wrapper of the SLEPc library through the JuliaSlepc.jl project.

## How to install it
You must have installed the PETSc library on your computer and set the two following environment variables : `PETSC_DIR` and `PETSC_ARCH`.

At run time, PetscWrap.jl looks for the `libpetsc.so` using these environment variables and "load" the library.

To install the package, use the Julia package manager:
```Julia
pkg> add PetscWrap
```
## Contribute
Any contribution(s) and/or remark(s) are welcome!

## PETSc compat.
This version of PetscWrap.jl has been tested with petsc-3.13. Complex numbers are not supported yet.

## How to use it
You will find examples of use by building the documentation: `julia PetscWrap.jl/docs/make.jl`. Here is one of the examples:
### A first demo
This example serves as a test since this project doesn't have a "testing" procedure yet. In this example,
the equation ``u'(x) = 2`` with ``u(0) = 0`` is solved on the domain ``[0,1]`` using a backward finite
difference scheme.

In this example, PETSc legacy method names are used. For more fancy names, check the next example.

Note that the way we achieve things in the document can be highly improved and the purpose of this example
is only demonstrate some method calls to give an overview.

To run this example, execute : `mpirun -n your_favorite_positive_integer julia example1.jl`

Import package

```julia
using PetscWrap
```

Initialize PETSc. Either without arguments, calling `PetscInitialize()` or using "command-line" arguments.
To do so, either provide the arguments as one string, for instance
`PetscInitialize("-ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always")` or provide each argument in
separate strings : `PetscInitialize(["-ksp_monitor_short", "-ksp_gmres_cgs_refinement_type", "refine_always")`

```julia
PetscInitialize()
```

Number of mesh points and mesh step

```julia
n = 11
Δx = 1. / (n - 1)
```

Create a matrix and a vector

```julia
A = MatCreate()
b = VecCreate()
```

Set the size of the different objects, leaving PETSC to decide how to distribute. Note that we should
set the number of preallocated non-zeros to increase performance.

```julia
MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
VecSetSizes(b, PETSC_DECIDE, n)
```

We can then use command-line options to set our matrix/vectors.

```julia
MatSetFromOptions(A)
VecSetFromOptions(b)
```

Finish the set up

```julia
MatSetUp(A)
VecSetUp(b)
```

Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
The `rstart, rend = *GetOwnershipRange` methods differ a little bit from PETSc API since the two integers it
returns are the effective Julia range of rows handled by the local processor. If `n` is the total
number of rows, then `rstart ∈ [1,n]` and `rend` is the last row handled by the local processor.

```julia
b_start, b_end = VecGetOwnershipRange(b)
```

Now let's build the right hand side vector. Their are various ways to do this, this is just one.

```julia
n_loc = VecGetLocalSize(b) ## Note that n_loc = b_end - b_start + 1...
VecSetValues(b, collect(b_start:b_end), 2 * ones(n_loc))
```

And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.

```julia
A_start, A_end = MatGetOwnershipRange(A)
for i in A_start:A_end
    A[i, i-1:i] = [-1. 1.] / Δx
end
```

Set boundary condition (only the proc handling index `1` is acting)

```julia
(b_start == 1) && (b[1] = 0.)
```

Assemble matrices

```julia
MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY)
VecAssemblyBegin(b)
MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY)
VecAssemblyEnd(b)
```

At this point, you can inspect `A` and `b` using the viewers (only stdout for now), simply call
`MatView(A)` and/or `VecView(b)`

Set up the linear solver

```julia
ksp = KSPCreate()
KSPSetOperators(ksp, A, A)
KSPSetFromOptions(ksp)
KSPSetUp(ksp)
```

Solve the system. We first allocate the solution using the `VecDuplicate` method.

```julia
x = VecDuplicate(b)
KSPSolve(ksp, b, x)
```

Print the solution

```julia
VecView(x)
```

Free memory

```julia
MatDestroy(A)
VecDestroy(b)
VecDestroy(x)
```

Finalize Petsc

```julia
PetscFinalize()

```



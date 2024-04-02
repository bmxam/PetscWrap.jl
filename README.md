[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://bmxam.github.io/PetscWrap.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://bmxam.github.io/PetscWrap.jl/dev)

# PetscWrap.jl

PetscWrap.jl is a parallel Julia wrapper for the (awesome) [PETSc](https://www.mcs.anl.gov/petsc/) library. It can be considered as a fork from the [GridapPetsc.jl](https://github.com/gridap/GridapPETSc.jl) and [Petsc.jl](https://github.com/JuliaParallel/PETSc.jl) projects : these two projects have extensively inspired this project, and some code has even been directly copied.

Note that the primary objective of this project is to enable the wrapper of the SLEPc library through the [SlepcWrap.jl](https://github.com/bmxam/SlepcWrap.jl) project.

This project is only a wrapper to PETSc functions, the purpose is not to deliver a julia `Array` (it maybe be one day the purpose of a package `PetscArrays.jl`).

## How to install it

To install the package, use the Julia package manager:

```Julia
pkg> add PetscWrap
```

If PETSc is not installed on your machine, it will be installed by the Julia package manager. Alternatively, if you already have a PETSc installation, `PetscWrap.jl` will select the install designated by `PETSC_DIR` and `PETSC_ARCH` environment variables.

If you want, at any time, to modify the PETSc install used by the wrapper, just type

```Julia
pkg> build PetscWrap
```

## Contribute

Any contribution(s) and/or remark(s) are welcome! If you need a function that is not wrapped yet but you don't think you are capable of contributing, post an issue with a minimum working example.

Conventions to be applied in future versions ("fancy" stuff is not concerned):

- all PETSc types should have the exact same name in Julia;
- all PETSc functions should have the exact same name in julia, but without the type as a prefix, and with a lower case for the first letter. `VecSetValues` becomes `setValues`. This rule is not applied when the name conflicts with a name from `Base` (for instance `VecView` becomes `vecView` and not `view`);
- all PETSc functions must have the same number of arguments and, if possible the same names in julia, except for out-of-place arguments.
- functions arguments must all be typed. Additional functions, without type or with fewer args, can be defined if the original version is present.

## PETSc compat.

This version of PetscWrap.jl has been tested with petsc-3.19. Complex numbers are supported.

## How to use it

PETSc methods wrappers share the almost same name as their C equivalent (with the type) : for instance `MatSetValues` becomes `setValues`. Furthermore, an optional "higher level" API, referred to as "fancy", is exposed : for instance `create_matrix` or `A[i,j] = v`. Note that this second way of manipulating PETSc will evolve according the package's author needs while the first one will try to follow PETSc official API.

You will find examples of use by building the documentation: `julia PetscWrap.jl/docs/make.jl`. Here is one of the examples:

### A first demo

This example serves as a test since this project doesn't have a "testing" procedure yet. In this example,
the equation `u'(x) = 2` with `u(0) = 0` is solved on the domain `[0,1]` using a backward finite
difference scheme.

In this example, PETSc classic method names are used. For more fancy names, check the fancy version.

Note that the way we achieve things in the document can be highly improved and the purpose of this example
is only demonstrate some method calls to give an overview.

To run this example, execute : `mpirun -n your_favorite_positive_integer julia example1.jl`

Import package

```julia
using PetscWrap
```

Initialize PETSc. Command line arguments passed to Julia are parsed by PETSc. Alternatively, you can
also provide "command line arguments by defining them in a string, for instance
`PetscInitialize("-ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always")` or by providing each argument in
separate strings : `PetscInitialize(["-ksp_monitor_short", "-ksp_gmres_cgs_refinement_type", "refine_always")`

```julia
PetscInitialize()
```

Number of mesh points and mesh step

```julia
n = 11
Δx = 1.0 / (n - 1)
```

Create a matrix and a vector (you can specify the MPI communicator if you want)

```julia
A = create(Mat)
b = create(Vec)
```

Set the size of the different objects, leaving PETSC to decide how to distribute. Note that we should
set the number of preallocated non-zeros to increase performance.

```julia
setSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
setSizes(b, PETSC_DECIDE, n)
```

We can then use command-line options to set our matrix/vectors.

```julia
setFromOptions(A)
setFromOptions(b)
```

Finish the set up

```julia
setUp(A)
setUp(b)
```

Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
As in PETSc, the `rstart, rend = *GetOwnershipRange` methods indicate the first row handled by the local processor
(starting at 0), and the last row (which is `rend-1`). This may be disturbing for non-regular PETSc users. Checkout
the fancy version of this example for a more Julia-like convention.

```julia
b_start, b_end = getOwnershipRange(b)
```

Now let's build the right hand side vector. Their are various ways to do this, this is just one.

```julia
n_loc = getLocalSize(b) # Note that n_loc = b_end - b_start...
setValues(b, collect(b_start:(b_end - 1)), 2 * ones(n_loc))
```

And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.

```julia
A_start, A_end = getOwnershipRange(A)
for i = A_start:(A_end - 1)
    setValues(A, [i], [i - 1, i], [-1.0 1.0] / Δx, INSERT_VALUES) # setValues(A, I, J, V, INSERT_VALUES)
end
```

Set boundary condition (only the proc handling index `0` is acting)

```julia
(b_start == 0) && setValue(b, 0, 0.0)
```

Assemble matrices

```julia
assemblyBegin(A, MAT_FINAL_ASSEMBLY)
assemblyBegin(b)
assemblyEnd(A, MAT_FINAL_ASSEMBLY)
assemblyEnd(b)
```

At this point, you can inspect `A` or `b` using a viewer (stdout by default), simply call

```julia
matView(A)
vecView(b)
```

Set up the linear solver

```julia
ksp = create(KSP)
setOperators(ksp, A, A)
setFromOptions(ksp)
setUp(ksp)
```

Solve the system. We first allocate the solution using the `VecDuplicate` method.

```julia
x = duplicate(b)
solve(ksp, b, x)
```

Print the solution

```julia
vecView(x)
```

Access the solution (this part is under development), getting a Julia array; and then restore it

```julia
array, ref = getArray(x) # do something with array
@show array
restoreArray(x, ref)
```

Free memory. Note that this call is faculative since, by default,
the julia GC will trigger a call to Petsc `destroy` to each object

```julia
destroy.((ksp, A, b, x))
```

Finalize Petsc. This call is faculative : it will be triggered automatically at the end of the script.

```julia
PetscFinalize()

```

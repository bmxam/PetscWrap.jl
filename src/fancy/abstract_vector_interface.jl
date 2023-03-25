# Implementation of AbstractArray interface for PetscVec

## Methods to implement
# size(A)
# getindex(A, i::Int)
# getindex(A, I::Vararg{Int, N})
# setindex!(A, v, i::Int)
# setindex!(A, v, I::Vararg{Int, N})

Base.size(vec::PetscVec) = (VecGetLocalSize(vec),) # Discutable choix : local or global size?

"""
`i` is 1-based, respecting Julia convention
"""
function Base.getindex(vec::PetscVec, i::Int)
    array, array_ref = VecGetValues(vec, PetscInt(1), PetscInt[i-1])
    return array[1]
end
Base.getindex(vec::PetscVec, I::Vararg{Int,N}) where {N} = error("method `getindex(A, I::Vararg{Int, N})` not implemented for PetscVec")

"""
`i` is 1-based, respecting Julia convention

# Implementation
For some unkwnown reason, calling `VecSetValue` fails.
"""
Base.setindex!(vec::PetscVec, v, i::Int) = VecSetValues(vec, PetscInt[i.-1], PetscScalar[v], INSERT_VALUES)
Base.setindex!(vec::PetscVec, v, I::Vararg{Int,N}) where {N} = error("method `setindex!(A, v, I::Vararg{Int, N})` not implemented for PetscVec")


## Optional methods
# IndexStyle(::Type)
# getindex(A, I...)
# setindex!(A, X, I...)
# iterate
# length(A)
# similar(A)
# similar(A, ::Type{S})
# similar(A, dims::Dims)
# similar(A, ::Type{S}, dims::Dims)

"""
`I` is 1-based, respecting Julia convention
"""
Base.setindex!(vec::PetscVec, X, I) = VecSetValues(vec, collect(I .- 1), X, INSERT_VALUES)

## Non-traditional indices
# axes(A)
# similar(A, ::Type{S}, inds)
# similar(T::Union{Type,Function}, inds)

## Misc :

# Discutable choice(s)
# Base.length(vec::PetscVec) = VecGetLocalSize(vec)

# Base.ndims(::Type{PetscVec}) = 1

# function Base.:*(x::PetscVec, alpha::Number)
#     y = VecCopy(x)
#     scale!(y, alpha)
#     return y
# end
# Base.:*(alpha::Number, vec::PetscVec) = vec * alpha

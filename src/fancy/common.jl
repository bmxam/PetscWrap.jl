destroy!(obj::Vararg{<:AbstractPetscObject,N}) where {N} = destroy.(obj)
set_up!(obj::Vararg{<:AbstractPetscObject,N}) where {N} = setUp.(obj)
set_from_options!(obj::Vararg{<:AbstractPetscObject,N}) where {N} = setFromOptions.(obj)

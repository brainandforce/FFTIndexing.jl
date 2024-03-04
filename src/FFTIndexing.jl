module FFTIndexing

import Base: length, size, axes, getindex, convert, to_index, to_indices
import Base: Tuple, OneTo, CartesianIndices, IndexStyle

# TODO: should this subtype Base.AbstractCartesianIndex{D}?
"""
    AbstractFFTIndex{D}

Supertype for array indices that correspond to the frequency bins constructed by performing a
Fourier transform of an array. This includes `FFTIndex{D}` and `NormalizedFFTIndex{D}`.

# Usage

`AbstractFFTIndex{D}` serves as an index for any object indexable by `CartesianIndex{D}`. Although
there is no direct method of converting an `AbstractFFTIndex{D}` to a `CartesianIndex{D}`, as that
requires information about the array being indexed, instances of this type can be used as an

Indexing an array with an `AbstractFFTIndex{D}` is guaranteed to be an inbounds access.

# Interface

For types `T<:AbstractFFTIndex{D}`:
  * The length of an `AbstractFFTIndex{D}` is always `D`.
  * The type constructors should accept a `Tuple{Vararg{Integer}}`. You will likely want to define
constructors that infer type parameters from the input. This will automatically define
`T(x::Vararg{Integer,D})`.
  * `Tuple(::T)` should be defined, returning the index as a `NTuple{D,Int}`.
  * `convert(::Type{<:FFTIndex}, ::T)` should be defined. This is to allow for types that encode
information about normalization factors to be converted to an `FFTIndex` used by generic indexing
methods. This will also automatically define `(::Type{<:FFTIndex})(::T)`.

Like `CartesianIndex{D}` in Julia 1.10 and up, `AbstractFFTIndex{D}` is a scalar type. To iterate
through its components, convert it to a `Tuple` with the `Tuple` constructor.
"""
abstract type AbstractFFTIndex{D}
end

(::Type{T})(x...) where T<:AbstractFFTIndex = T(x)

length(::Type{<:AbstractFFTIndex{D}}) where D = D
length(x::AbstractFFTIndex) = length(typeof(x))

getindex(x::AbstractFFTIndex, i::Integer) = (@boundscheck checkbounds(x, i); return x.I[i])

# This one's stolen from Julia Base
function Base.iterate(::T) where T<:AbstractFFTIndex
    error(
        "Like CartesianIndex, iteration is deliberately unsupported for $T. " * 
        "Use `I` rather than `I...`, or use `Tuple(I)...`"
    )
end

Tuple(x::AbstractFFTIndex) = x.I
convert(::Type{T}, x::AbstractFFTIndex) where T<:Tuple = x.I

"""
    FFTIndex{D} <: AbstractFFTIndex{D}

A index for an array corresponding to an FFT frequency bin along each dimension of an array. These
frequency bins are integer values which are easily converted to corresponding array indices.
"""
struct FFTIndex{D} <: AbstractFFTIndex{D}
    I::NTuple{D,Int}
end

FFTIndex(t::Tuple{Vararg{Integer,D}}) where D = FFTIndex{D}(t)

Base.show(io::IO, i::FFTIndex) = print(io, FFTIndex, Tuple(i))

#---Machinery for other AbstractFFTIndex features--------------------------------------------------#

(T::Type{<:FFTIndex})(i::AbstractFFTIndex{D}) where D = convert(T, i)

"""
    FFTIndex._cartesian_tuple(i::AbstractFFTIndex, ax::Tuple)

Generates a `Tuple` which may be used to construct a `CartesianIndex` that performs the same
indexing as `i`. This function is provided separately in case the `CartesianIndex` does not need to
be constructed and a raw `Tuple` is desired.
"""
function _cartesian_tuple(i::AbstractFFTIndex, ax::Tuple)
    length(i) === length(ax) || throw(
        DimensionMismatch(
            "The length of the AbstractFFTIndex must match the number of axes.\n" *
            "If you are trying to construct a CartesianIndex for specific axes, the axes must be " *
            "explicitly specified:\n" *
            "\tCartesianIndex(i, axes(A)[...])    # for indexable object A\n" *
            "\tCartesianIndex(i, ax[...])         # for list of axes ax"
        )
    )
    return map((x, t) -> mod(x, length(t)) + first(t), Tuple(FFTIndex(i)), ax)
end

"""
    CartesianIndex(i::AbstractFFTIndex, inds::Tuple)
    CartesianIndex(i::AbstractFFTIndex, A)

Constructs a `CartesianIndex` corresponding to the the index `i` using information from `A` or a set
of axes `ax`.
"""
(::Type{T})(i::AbstractFFTIndex, ax::Tuple) where T<:CartesianIndex = T(_cartesian_tuple(i, ax))
(::Type{T})(i::AbstractFFTIndex, A) where T<:CartesianIndex = CartesianIndex(i, axes(A))

to_indices(A, I::Tuple{AbstractFFTIndex,Vararg}) = to_indices(A, axes(A), I)

function to_indices(A, inds, I::Tuple{AbstractFFTIndex{D},Vararg}) where D
    return (_cartesian_tuple(first(I), inds[1:D])..., to_indices(A, inds[D:end], I[2:end])...)
end

# Remove bounds checks when only indexing with AbstractFFTIndex
function getindex(A::AbstractArray, i::AbstractFFTIndex...)
    return @inbounds A[map(x -> CartesianIndex(x, A), i)...]
end

getindex(t::Tuple, i::AbstractFFTIndex{1}) = @inbounds t[CartesianIndex(i, only(axes(t)))]

#---FFT index normalized by array size-------------------------------------------------------------#
#=
"""
    NormalizedFFTIndex{D} <: AbstractFFTIndex{D}

An index for an array similar to `FFTIndex{D}`, but simultaneously including a normalization factor
calculated from the length of each dimension.
"""
struct NormalizedFFTIndex{D} <: AbstractFFTIndex{D}
    # I::NTuple{D,Rational{Int}} can't be used because it simplifies!
    inds::FFTIndex{D}
    size::NTuple{D,Int}
end

NormalizedFFTIndex(t::Tuple{Vararg{Real,D}}) where D = NormalizedFFTIndex{D}(t)

(::Type{T})(A, i::FFTIndex) where T<:NormalizedFFTIndex = i.inds

convert(::Type{<:FFTIndex}, i::NormalizedFFTIndex{D}) where D = i.inds

function to_index(A, i::NormalizedFFTIndex)
    factors = denominator.(i.I)
    factors === size(A) || @warn(
        string(
            "Normalization factors do not match the array dimensions:\n" * 
            "Index indicates size $factors; size of the indexed object is $(size(A))."
        )
    )
    return to_index(A, FFTIndex(i))
end
=#
#---FFTAxis{D}-------------------------------------------------------------------------------------#
"""
    FFTAxis <: AbstractVector{Int}

The one-dimensional counterpart to `FFTIndices`, supporting only a single dimension. Its elements
are plain `Int` types rather than the `CartesianIndex` of `FFTIndices`.

This type serves as an analog to `Base.OneTo` which is usually returned by `axes`; this package
provides `fftaxes` to serve the same purpose.

# Examples
```
julia> FFTAxis(4)
4-element FFTAxis:
 0
 1
 -2
 -1
```
"""
struct FFTAxis <: AbstractVector{Int}
    size::Int
    FFTAxis(x::Integer) = new(x)
end

FFTAxis(v::AbstractVector) = FFTAxis(length(v))

axes(r::FFTAxis) = (OneTo(r.size),)
size(r::FFTAxis) = (r.size,)
IndexStyle(::Type{<:FFTAxis}) = IndexLinear()

getindex(r::FFTAxis, i::Integer) = mod(i + div(length(r), 2) - 1, length(r)) - div(length(r), 2)
getindex(r::FFTAxis, i::CartesianIndex{1}) = r[only(i.I)]

Base.iterate(r::FFTAxis, i = 1) = i in eachindex(r) ? (r[i], i+1) : nothing

Base.minimum(r::FFTAxis) = -div(length(r), 2)
Base.maximum(r::FFTAxis) = div(length(r) - 1, 2)
# For Julia 1.6 compatibility: must use keyword arguments
Base.sort(r::FFTAxis) = range(minimum(r), stop = maximum(r))

"""
    fftaxes(a, d) -> FFTAxis

Returns an `FFTAxis` associated with an object `a` along dimension `d`. This is calculated from
`size(a, d)`.
"""
fftaxes(a, d) = FFTAxis(size(a, d))

"""
    fftaxes(a) -> NTuple{D,FFTAxis}

Returns a set of `FFTAxis` objects associated with an object `a`. This is calculated from `size(a)`.
"""
fftaxes(a) = FFTAxis.(size(a))

Base.show(io::IO, r::FFTAxis) = print(io, FFTAxis, '(', r.size, ')')

#---Analog to `CartesianIndices{D}`----------------------------------------------------------------#
"""
    FFTIndices{D} <: AbstractArray{FFTIndex{D},D}

An iterable object defining a range of CartesianIndices corresponding to FFT indices. This can be
used to convert a Cartesian array index to an FFT bin index, or vice versa.

An array of `FFTIndex{D}` objects which correspond to all valid indices of an indexable object.

The outputs use the convention where frequencies at or above the Nyquist frequency for that
dimension are negative, matching the output of `FFTW.fftfreq`.

# Examples
```
julia> FFTIndices(4)
4-element FFTIndices{1}:
 FFTIndex(0,)
 FFTIndex(1,)
 FFTIndex(-2,)
 FFTIndex(-1,)

julia> FFTIndices(3, 3)
3×3 FFTIndices{2}:
 FFTIndex(0, 0)   FFTIndex(0, 1)   FFTIndex(0, -1)
 FFTIndex(1, 0)   FFTIndex(1, 1)   FFTIndex(1, -1)
 FFTIndex(-1, 0)  FFTIndex(-1, 1)  FFTIndex(-1, -1)
```
"""
struct FFTIndices{D} <: AbstractArray{FFTIndex{D},D}
    axes::NTuple{D,FFTAxis}
end

FFTIndices(x::Tuple{Vararg{Integer,D}}) where D = FFTIndices{D}(FFTAxis.(x))
FFTIndices(x::Tuple{Vararg{UnitRange,D}}) where D = FFTIndices{D}(FFTAxis.(x))

FFTIndices(a) = FFTIndices(fftaxes(a))

size(r::FFTIndices) = length.(r.axes)
axes(r::FFTIndices) = OneTo.(size(r))
fftaxes(r::FFTIndices) = r.axes
fftaxes(r::FFTIndices{D}, d) where D = ifelse(d::Integer <= D, r.axes[d::Integer], FFTAxis(1))

function getindex(r::FFTIndices{D}, inds::Vararg{Int,D}) where D
    @boundscheck checkbounds(r, inds...)
    return FFTIndex(map((ax, i) -> ax[i], r.axes, inds))
end

Base.@propagate_inbounds getindex(r::FFTIndices, i::Int) = r[CartesianIndices(r)[i]]

# True by default: IndexStyle(::Type{<:FFTIndices}) = IndexCartesian()
# Base.iterate(r::FFTIndices, i = 1) = ifelse(i in eachindex(r), (r[i], i+1), nothing)

#---Exports----------------------------------------------------------------------------------------#

export AbstractFFTIndex, FFTIndex, #=NormalizedFFTIndex,=# FFTAxis, FFTIndices
export fftaxes

end

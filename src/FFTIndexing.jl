module FFTIndexing

using AbstractFFTs

import Base: length, size, axes, getindex, IndexStyle
import Base: Tuple, OneTo, CartesianIndices

# TODO: should this subtype Base.AbstractCartesianIndex{D}?
"""
    AbstractFFTIndex{D}

Supertype for `FFTIndex{D}` and `NormalizedFFTIndex{D}`.

# Interface

For types `T<:AbstractFFTIndex{D}`
  * The length of an `AbstractFFTIndex{D}` is always `D`.
  * `Tuple(::T)` should be defined.
  * `convert(::Type{<:FFTIndex}, ::T)` should be defined.
"""
abstract type AbstractFFTIndex{D}
end

(::Type{T})(x::Vararg{Integer,D}) where {D,T<:AbstractFFTIndex{D}} = T(x)

length(::Type{<:AbstractFFTIndex{D}}) where D = D
length(x::AbstractFFTIndex) = length(typeof(x))

getindex(x::AbstractFFTIndex, i::Integer) = (@boundscheck checkbounds(x, i); return x.I[i])

Tuple(x::AbstractFFTIndex) = x.I

"""
    FFTIndex{D} <: AbstractFFTIndex{D}

A index for an array corresponding to an FFT frequency bin along each dimension of an array.

# Use with arrays

Indexing an `AbstractArray{T,D}` with `FFTIndex{D}` is guaranteed to be an inbounds array access,
as the 
"""
struct FFTIndex{D} <: AbstractFFTIndex{D}
    I::NTuple{D,Int}
end

FFTIndex(x::Vararg{Integer,D}) where D = FFTIndex{D}(x)

(T::Type{<:FFTIndex})(x::AbstractFFTIndex{D}) where D = convert(T, x)

"""
    NormalizedFFTIndex{D}

An index for an array similar to `FFTIndex{D}`, but conveying information about the array size as
a normalization factor.
"""
struct NormalizedFFTIndex{D}
    I::NTuple{D,Rational{Int}}
end

NormalizedFFTIndex(x::Vararg{Integer,D}) where D = NormalizedFFTIndex{D}(x)

Base.convert(::Type{<:FFTIndex}, x::NormalizedFFTIndex{D}) where D = FFTIndex(numerator.(x.I))

#---Using FFTIndex{D}------------------------------------------------------------------------------#

"""
    FFTAxis <: AbstractVector{Int}

The one-dimensional counterpart to `FFTIndices`, supporting only a single dimension. Its elements
are plain `Int` types rather than the `CartesianIndex` of `FFTIndices`.

In essence, it serves as an analog to `Base.OneTo`.

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

function Base.getindex(r::FFTAxis, i::Integer)
    return mod(i + div(length(r), 2) - 1, length(r)) - div(length(r), 2)
end

getindex(r::FFTAxis, i::CartesianIndex{1}) = r[only(i.I)]

Base.iterate(r::FFTAxis, i = 1) = i in eachindex(r) ? (r[i], i+1) : nothing

# For Julia 1.6 compatibility: must use keyword arguments
Base.sort(r::FFTAxis) = range(minimum(r), stop = maximum(r))

"""
    fftaxes(a::AbstractArray{T,D}) -> NTuple{D,FFTAxis}

Returns a set of `FFTAxis` objects associated with an array `a`.
"""
fftaxes(a::AbstractArray) = FFTAxis.(size(a))

#---Analog to `CartesianIndices{D}`----------------------------------------------------------------#
"""
    FFTIndices{D} <: AbstractArray{CartesianIndex{D},D}

An iterable object defining a range of CartesianIndices corresponding to FFT indices. This can be
used to convert a Cartesian array index to an FFT bin index, or vice versa.

The outputs use the convention where frequencies at or above the Nyquist frequency for that
dimension are negative, matching the output of `FFTW.fftfreq`.

# Examples
```
julia> FFTIndices(4)
4-element FFTIndices{1}:
 CartesianIndex(0,)
 CartesianIndex(1,)
 CartesianIndex(-2,)
 CartesianIndex(-1,)

julia> FFTIndices(3, 3)
3×3 FFTIndices{2}:
 CartesianIndex(0, 0)   CartesianIndex(0, 1)   CartesianIndex(0, -1)
 CartesianIndex(1, 0)   CartesianIndex(1, 1)   CartesianIndex(1, -1)
 CartesianIndex(-1, 0)  CartesianIndex(-1, 1)  CartesianIndex(-1, -1)
```
"""
struct FFTIndices{D} <: AbstractArray{FFTIndex{D},D}
    size::NTuple{D,Int}
end

FFTIndices(x::Tuple{Vararg{Integer,D}}) where D = FFTIndices{D}(Int.(x))
FFTIndices(x::Tuple{Vararg{UnitRange,D}}) where D = FFTIndices{D}(length.(x))

FFTIndices(a::AbstractArray) = FFTIndices(size(a))

axes(r::FFTIndices) = OneTo.(r.size)
size(r::FFTIndices) = r.size
IndexStyle(::Type{<:FFTIndices}) = IndexLinear()

function Base.getindex(r::FFTIndices{D}, i::CartesianIndex{D}) where D
    return CartesianIndex(mod.(Tuple(i) .+ div.(size(r), 2) .- 1, size(r)) .- div.(size(r), 2))
end

getindex(r::FFTIndices, i::Integer...) = r[CartesianIndex(i)]
getindex(r::FFTIndices, i::Integer) = r[CartesianIndices(r)[i]]

Base.iterate(r::FFTIndices, i = 1) = ifelse(i in eachindex(r), (r[i], i+1), nothing)

#---Exports----------------------------------------------------------------------------------------#

export AbstractFFTIndex, NormalizedFFTIndex, FFTAxis, FFTIndices
export fftaxes

end

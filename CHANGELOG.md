All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
  - Support for the `rev` keyword in `sort(::FFTAxis)`.

## [0.1.1] - 2024-03-10

### Added
  - [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) for automated testing.

### Fixed
  - All indexing with `FFTIndices` objects is properly implemented and tested.
  - Resolved a method ambiguity with `(::Type{<:FFTIndex})(::AbstractFFTIndex)` constructor.

### Removed
  - Special `getindex` methods for `AbstractArray`.

## [0.1.0] - 2024-03-04

### Added
  - Abstract types: `AbstractFFTIndex{D}`
  - Data structures: `FFTIndex{D}`, `FFTAxis`, and `FFTIndices{D}`
  - Functions: `fftaxes`

[Unreleased]: https://github.com/brainandforce/FFTIndexing.jl
[0.1.1]: https://github.com/brainandforce/FFTIndexing.jl/releases/tag/v0.1.1
[0.1.0]: https://github.com/brainandforce/FFTIndexing.jl/releases/tag/v0.1.0

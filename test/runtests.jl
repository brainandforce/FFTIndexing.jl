using FFTIndexing
using Test

@testset "All tests" begin
    @testset "AbstractFFTIndex" begin
        @test FFTIndex <: AbstractFFTIndex
        # @test NormalizedFFTIndex <: AbstractFFTIndex
        A = [m*n for m in 1:6, n in 1:9]
        c1 = CartesianIndex(3,4)
        i1 = FFTIndex{2}((2,3))
        # n1 = NormalizedFFTIndex{2}((2//6, 3//9))
        @test i1 === FFTIndex{2}(2,3)
        @test i1 === FFTIndex((2,3))
        @test i1 === FFTIndex(2,3)
        @test A[i1] === A[CartesianIndex(3,4)]
        # @test n1 === NormalizedFFTIndex{2}(2//6, 3//9)
        # @test n1 === NormalizedFFTIndex((2//6, 3//9))
        # @test n1 === NormalizedFFTIndex(2//6, 3//9)
        # @test n1 === NormalizedFFTIndex{2}(A, i1)
        # @test n1 === NormalizedFFTIndex(A, i1)
        # @test A[n1] === A[CartesianIndex(3,4)]
        B = reshape(1:63, 3, 3, 7)
        i2 = FFTIndex(1, 5)
        @test_throws DimensionMismatch CartesianIndex(i2, B)
        @test CartesianIndex(i2, axes(B)[1:2]) === CartesianIndex(2, 3)
        @test B[i2, 4] === B[CartesianIndex(i2, axes(B)[1:2]), 4]
        @test (2, 's')[FFTIndex(0)] === 2
        @test (2, 's')[FFTIndex(1)] === 's'
    end
    @testset "FFTAxes" begin
        A = [m*n for m in 1:6, n in 1:9]
        @test all(FFTAxis(4) .== [0, 1, -2, -1])
        @test all(FFTAxis(5) .== [0, 1, 2, -2, -1])
        @test length(FFTAxis(4)) === 4
        @test length(FFTAxis(5)) === 5
        @test sort(FFTAxis(4)) == -2:1
        @test sort(FFTAxis(5)) == -2:2
        @test fftaxes(A) === (FFTAxis(6), FFTAxis(9))
        @test fftaxes(A, 1) === FFTAxis(6)
        @test fftaxes(A, 2) === FFTAxis(9)
        @test eval(Meta.parse(repr(fftaxes(A)))) === (FFTAxis(6), FFTAxis(9))
    end
    @testset "FFTIndices" begin
        A = [m*n for m in 1:6, n in 1:9]
        @test FFTIndices(A) === FFTIndices((1:6, 1:9))
        @test FFTIndices(A) === FFTIndices((6, 9))
        @test size(FFTIndices(A)) === size(A)
        @test length(FFTIndices(A)) === length(A)
        @test fftaxes(FFTIndices(A)) === fftaxes(A)
        @test fftaxes(FFTIndices(A)) === FFTIndices(A).axes
        @test A[FFTIndices(A)[42]] === A[42]
        @test A[FFTIndices(A)[42]] === A[CartesianIndices(A)[42]]
        @test LinearIndices(FFTIndices(A)) === LinearIndices(CartesianIndices(A))
    end
end

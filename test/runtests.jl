using FFTIndexing
using Test

@testset "All tests" begin
    @testset "AbstractFFTIndex" begin
        @test FFTIndex <: AbstractFFTIndex
        @test NormalizedFFTIndex <: AbstractFFTIndex
        A = [m*n for m in 1:6, n in 1:9]
        c1 = CartesianIndex(3,4)
        i1 = FFTIndex{2}((2,3))
        n1 = NormalizedFFTIndex{2}((2//6, 3//9))
        @test i1 === FFTIndex{2}(2,3)
        @test i1 === FFTIndex((2,3))
        @test i1 === FFTIndex(2,3)
        @test n1 === NormalizedFFTIndex{2}(2//6, 3//9)
        @test n1 === NormalizedFFTIndex((2//6, 3//9))
        @test n1 === NormalizedFFTIndex(2//6, 3//9)
        @test n1 === NormalizedFFTIndex{2}(A, i1)
        @test n1 === NormalizedFFTIndex(A, i1)
        @test A[i1] === A[CartesianIndex(3,4)]
        @test A[n1] === A[CartesianIndex(3,4)]
    end
end

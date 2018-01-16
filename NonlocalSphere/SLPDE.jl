abstract type LinearOperator{T} end

struct NonlinearOperator{F}
    N::F
end

struct SemiLinearPDE{D, T, LL, NN}
    L::LL
    N::NN
    U0::NTuple{D, Matrix{T}}
    function SemiLinearPDE{D, T, LL, NN}(L::LL, N::NN, U0::NTuple{D, Matrix{T}}) where {D, T, LL, NN}
        if D > 1
            @assert LL <: Tuple
            @assert NN <: Tuple
            @assert length(L) == D
            @assert length(N) == D
        end
        new{D, T, LL, NN}(L, N, U0)
    end
end

SemiLinearPDE(L, N, U0::NTuple{D, Matrix{T}}) where {D, T} = SemiLinearPDE{D, T, typeof(L), typeof(N)}(L, N, U0)
SemiLinearPDE(L, N, U0::Matrix{T}) where {T} = SemiLinearPDE(L, N, (U0,))

# ϵ^2 Δ
struct LaplaceBeltrami{T} <: LinearOperator{T}
    ϵ::T
end

# ϵ^2 ℒ_{δ}(α)
struct NonlocalLaplaceBeltrami{T} <: LinearOperator{T}
    ϵ::T
    α::T
    δ::T
end

function create_linear_operator(LB::LaplaceBeltrami{T}, m::Integer, n::Integer) where T
    L = zeros(T, m, 2n-1)
    ϵ = LB.ϵ
    @inbounds for i = 1:m
        L[i,1] = -ϵ^2*i*(i-1)
    end
    @inbounds for j = 1:n-1
        for i = 1:m-j
            L[i,2j] = L[i+j,1]
            L[i,2j+1] = L[i+j,1]
        end
    end
    L
end

function create_linear_operator(NLB::NonlocalLaplaceBeltrami{T}, m::Integer, n::Integer) where T
    L = zeros(T, m, 2n-1)
    ϵ = NLB.ϵ
    @inbounds for i = 1:m
        L[i,1] = ϵ^2*evaluate_lambda(i-1, NLB.δ, NLB.α)
    end
    @inbounds for j = 1:n-1
        for i = 1:m-j
            L[i,2j] = L[i+j,1]
            L[i,2j+1] = L[i+j,1]
        end
    end
    L
end

function evaluate_nonlinear_operator!(NL::NonlinearOperator, P, Ps, Pa, U::Matrix{T}, V::Matrix{T}, F::Matrix{T}, NF::Matrix{T}, NV::Matrix{T}, NU::Matrix{T}) where T
    # Compute spherical harmonic coefficients of a nonlinear operator N(u).

    # sph2fourier
    A_mul_B!(fill!(V, zero(T)), P, U)
    # Fourier to function values on sphere
    A_mul_B!(fill!(F, zero(T)), Ps, V)
    NF .= NL.N.(F)
    # Function values on sphere to Fourier
    A_mul_B!(fill!(NV, zero(T)), Pa, NF)
    # fourier2sph
    At_mul_B!(fill!(NU, zero(T)), P, NV)

    return NU
end

function evaluate_nonlinear_operator!(NL::Tuple{NonlinearOperator{F1}, NonlinearOperator{F2}}, P, Ps, Pa, U::NTuple{2, Matrix{T}}, V::NTuple{2, Matrix{T}}, F::NTuple{2, Matrix{T}}, NF::NTuple{2, Matrix{T}}, NV::NTuple{2, Matrix{T}}, NU::NTuple{2, Matrix{T}}) where {T, F1, F2}
    # Compute spherical harmonic coefficients of a nonlinear operator N(u).

    # sph2fourier
    A_mul_B!(fill!(V[1], zero(T)), P[1], U[1])
    A_mul_B!(fill!(V[2], zero(T)), P[2], U[2])
    # Fourier to function values on sphere
    A_mul_B!(fill!(F[1], zero(T)), Ps[1], V[1])
    A_mul_B!(fill!(F[2], zero(T)), Ps[2], V[2])
    NF[1] .= NL[1].N.(F[1], F[2])
    NF[2] .= NL[2].N.(F[1], F[2])
    # Function values on sphere to Fourier
    A_mul_B!(fill!(NV[1], zero(T)), Pa[1], NF[1])
    A_mul_B!(fill!(NV[2], zero(T)), Pa[2], NF[2])
    # fourier2sph
    At_mul_B!(fill!(NU[1], zero(T)), P[1], NV[1])
    At_mul_B!(fill!(NU[2], zero(T)), P[2], NV[2])

    return NU
end

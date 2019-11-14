import FastTransforms: FTPlan

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
    λ = evaluate_lambda(m, NLB.α, NLB.δ)
    ϵ = NLB.ϵ
    @inbounds for i = 1:m
        L[i,1] = ϵ^2*λ[i]
    end
    @inbounds for j = 1:n-1
        for i = 1:m-j
            L[i,2j] = L[i+j,1]
            L[i,2j+1] = L[i+j,1]
        end
    end
    L
end

function evaluate_nonlinear_operator!(NL::NonlinearOperator, P::FTPlan{T, 2, FastTransforms.SPHERE}, PS::FTPlan{T, 2, FastTransforms.SPHERESYNTHESIS}, PA::FTPlan{T, 2, FastTransforms.SPHEREANALYSIS}, NU::Matrix{T}) where T
    # Compute spherical harmonic coefficients of a nonlinear operator N(u).

    # sph2fourier
    lmul!(P, NU)
    # Fourier to function values on sphere
    lmul!(PS, NU)
    NU .= NL.N.(NU)
    # Function values on sphere to Fourier
    lmul!(PA, NU)
    # fourier2sph
    ldiv!(P, NU)

    return NU
end

function evaluate_nonlinear_operator!(NL::Tuple{NonlinearOperator{F1}, NonlinearOperator{F2}}, P::FTPlan{T, 2, FastTransforms.SPHERE}, PS::FTPlan{T, 2, FastTransforms.SPHERESYNTHESIS}, PA::FTPlan{T, 2, FastTransforms.SPHEREANALYSIS}, NU::NTuple{2, Matrix{T}}) where {T, F1, F2}
    # Compute spherical harmonic coefficients of a nonlinear operator N(u).

    # sph2fourier
    lmul!(P, NU[1])
    lmul!(P, NU[2])
    # Fourier to function values on sphere
    lmul!(PS, NU[1])
    lmul!(PS, NU[2])
    NU[1], NU[2] .= NL[1].N.(NU[1], NU[2]), NL[2].N.(NU[1], NU[2])
    # Function values on sphere to Fourier
    lmul!(PA, NU[1])
    lmul!(PA, NU[2])
    # fourier2sph
    ldiv!(P, NU[1])
    ldiv!(P, NU[2])

    return NU
end

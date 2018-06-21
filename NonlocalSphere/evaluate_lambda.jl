using FastTransforms

import FastTransforms: RecurrencePlan, backward_recurrence, chebyshevjacobimoments1, clenshawcurtisweights!

#
# Numerical evaluation of P_n(cospi(θ)) - 1 via an expansion in half-angled
# sines. This should only be effective between θ = 0 and the nearest root.
# Beyond this, it would be unstable, but recurrence relations or
# asymptotics are fine.
#
function legendre_minus_one(n::Int, θ::Number)
    spθ2 = sinpi(θ/2)^2
    ret = zero(θ)
    binom = one(θ)
    spθ2k = one(θ)
    for k = 1:n
        binom *= -(n+one(θ)-k)*(n+k)/k^2
        spθ2k *= spθ2
        cst = binom*spθ2k
        if !isfinite(cst) break end
        ret += cst
    end

    ret
end

function legendre_asy(n::Int, θ::Number)
    if θ > 1/2
        return iseven(n) ? legendre_asy(n, 1-θ) : -legendre_asy(n, 1-θ)
    end
    πθ = π*θ
    ret = zero(θ)
    spθ = sinpi(θ)
    cpθ = cospi(θ)
    nphalf = n+one(θ)/2
    arg = nphalf*πθ

    J0 = besselj(0, arg)
    ret += J0

    J1 = besselj(1, arg)
    nphalfν = nphalf
    ret += (πθ*cpθ-spθ)/(8*πθ*spθ)*J1/nphalfν

    J2 = 2/arg*J1-J0
    nphalfν *= nphalf
    ret += (6*πθ*spθ*cpθ-15spθ^2+πθ^2*(9-spθ^2))/(128*πθ^2*spθ^2)*J2/nphalfν

    J3 = 4/arg*J2-J1
    nphalfν *= nphalf
    ret += 5*(((πθ^3+21πθ)*spθ^2+15πθ^3)*cpθ-((3πθ^2+63)*spθ^2-27πθ^2)*spθ)/(1024*πθ^3*spθ^3)*J3/nphalfν

    ret *= sqrt(θ*π/sinpi(θ))
end

#
# Numerical evaluation of the integral:
#
# (1+α) 2^(2-α)/δ^2 ∫_{-1}^1 (P_ℓ(1-δ^2/2*((1-x)/2))-1)/(1-x) * (1-x)^α dx,
#
# by Jacobi-weighted Clenshaw-Curtis quadrature.
#
function evaluate_lambda{T}(ℓ::Int, δ::T, α::T)
    if iszero(δ)
        return -ℓ*(ℓ+one(T))
    elseif ℓ == 0
        return zero(T)
    elseif ℓ ≤ 50
        return (1+α)*(T(2))^(2-α)/δ^2*evaluate_lambda_rec(ℓ, δ, α)
    else
        return (1+α)*(T(2))^(2-α)/δ^2*evaluate_lambda_asy(ℓ, δ, α)
    end
end

evaluateLambda(ℓ, δ, α) = evaluate_lambda(round(Int, ℓ), δ, α)

function evaluate_lambda_rec{T}(ℓ::Int, δ::T, α::T)
    # Clenshaw-Curtis points in angle (mod pi)
    θ = T[k/ℓ for k in zero(T) : ℓ]
    w = clenshawcurtisweights!(chebyshevjacobimoments1(T, ℓ+1, α, zero(T)))
    φ = T[2/π*asin(δ/2*sinpi(θ/2)) for θ in θ]

    RP = RecurrencePlan(zero(T), zero(T), ℓ+1)
    c = T[zeros(T, ℓ); 1]

    s = zero(T)
    for k in 1:ℓ+1
        if φ[k] == 0
            s += -w[k]*δ^2/4*ℓ*(ℓ+1)/2
        elseif φ[k] < 1/2ℓ # A bound on the first root of the Legendre polynomial.
            s += w[k]*legendre_minus_one(ℓ, φ[k])/(2*sinpi(θ[k]/2)^2)
        else
            s += w[k]*(backward_recurrence(c, φ[k], RP)-1)/(2*sinpi(θ[k]/2)^2)
        end
    end

    return s
end

function evaluate_lambda_asy{T}(ℓ::Int, δ::T, α::T)
    # Clenshaw-Curtis points in angle (mod pi)
    θ = T[k/ℓ for k in zero(T) : ℓ]
    w = clenshawcurtisweights!(chebyshevjacobimoments1(T, ℓ+1, α, zero(T)))
    φ = T[2/π*asin(δ/2*sinpi(θ/2)) for θ in θ]

    s = zero(T)
    for k in 1:ℓ+1
        if φ[k] == 0
            s += -w[k]*δ^2/4*ℓ*(ℓ+1)/2
        elseif φ[k] == 1
            s += w[k]*(iseven(ℓ) ? zero(T) : -one(T))
        elseif φ[k] < 1/2ℓ # A bound on the first root of the Legendre polynomial.
            s += w[k]*legendre_minus_one(ℓ, φ[k])/(2*sinpi(θ[k]/2)^2)
        else
            s += w[k]*(legendre_asy(ℓ, φ[k])-1)/(2*sinpi(θ[k]/2)^2)
        end
    end

    return s
end

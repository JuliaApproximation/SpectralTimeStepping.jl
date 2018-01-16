function sphcesaro1(A::AbstractMatrix{T}) where T
    M, N = size(A)
    m, n = M, N÷2
    C = zeros(T, m, 2n+1)
    @inbounds for i = 1:m
        C[i,1] = (m+one(T)-i)/m
    end
    @inbounds for j = 1:n
        for i = 1:m-j
            C[i,2j] = C[i+j,1]
            C[i,2j+1] = C[i+j,1]
        end
    end
    C
end

function sphcesaro2(A::AbstractMatrix{T}) where T
    M, N = size(A)
    m, n = M, N÷2
    C = zeros(T, m, 2n+1)
    @inbounds for i = 1:m
        C[i,1] = (m+2one(T)-i)*(m+one(T)-i)/((m+one(T))*m)
    end
    @inbounds for j = 1:n
        for i = 1:m-j
            C[i,2j] = C[i+j,1]
            C[i,2j+1] = C[i+j,1]
        end
    end
    C
end


function sphcesaro(A::AbstractMatrix{T}, κ::Int) where T
    M, N = size(A)
    m, n = M, N÷2
    C = zeros(T, m, 2n+1)
    @inbounds for i = 1:m
        C[i,1] = binomial(m+κ-i, m-i)/binomial(m+κ-1, m-1)
    end
    @inbounds for j = 1:n
        for i = 1:m-j
            C[i,2j] = C[i+j,1]
            C[i,2j+1] = C[i+j,1]
        end
    end
    C
end

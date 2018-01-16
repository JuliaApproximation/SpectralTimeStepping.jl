include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

using PlotlyJS

n = 512

θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

ϕ = 1/8
c = cospi(ϕ)
s = sinpi(ϕ)

F = (cos.(40.0.*(c.*x.-s.*z)).+cos.(40.0.*y).+cos.(40.0.*(s.*x.+c.*z)))./3
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
U0 = fourier2sph(V; sketch = :none), zero(F)

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
F = [F F[:,1]]

s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = -1.0, cmax = 1.0, showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/LGLu0.pdf";js=:remote)
savefig(p,"/Users/Mikael/NGLu0.pdf";js=:remote)


IMAX = 4

SLPDE = SemiLinearPDE((LaplaceBeltrami(0.01),LaplaceBeltrami(0.01)), (NonlinearOperator((u,v)->u-(u-1.5v)*(u^2+v^2)),NonlinearOperator((u,v)->v-(v+1.5u)*(u^2+v^2))), U0)
UL = Vector{NTuple{2,Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    T = 1.0*4^(i-1)
    n = round(Int, 10*T)
    UL[i] = ETDRK4(SLPDE, T, n)
    println("Done i = ", i)

    V = sph2fourier(UL[i][1]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = -1.0, cmax = 1.0, showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/LGLu$(i).pdf";js=:remote)
end

SLPDE = SemiLinearPDE((NonlocalLaplaceBeltrami(0.01, 0.0, 1.0),NonlocalLaplaceBeltrami(0.01, 0.0, 1.0)), (NonlinearOperator((u,v)->u-(u-1.5v)*(u^2+v^2)),NonlinearOperator((u,v)->v-(v+1.5u)*(u^2+v^2))), U0)
UN = Vector{NTuple{2,Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    T = 1.0*4^(i-1)
    n = round(Int, 10*T)
    UN[i] = ETDRK4(SLPDE, T, n)
    println("Done i = ", i)

    V = sph2fourier(sphcesaro2(UN[i][1]).*UN[i][1]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = -1.0, cmax = 1.0, showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/NGLu$(i).pdf";js=:remote)
end

include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

using PlotlyJS

n = 100

θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

RHSF = -exp.(-30.*((x-1/4).^2+(y-sqrt(11)/4).^2+(z-1/2).^2))-exp.(-50.*z.^2)

RHSV = zero(RHSF)
A_mul_B!(RHSV, FastTransforms.plan_analysis(RHSF), RHSF)
RHSU = fourier2sph(RHSV)


θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

RHSF = [mean(RHSF[1,:])*ones(1, size(RHSF, 2)); RHSF; mean(RHSF[end,:])*ones(1, size(RHSF, 2))]
RHSF = [RHSF RHSF[:,1]]

s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = RHSF, cmin = minimum(RHSF), cmax = maximum(RHSF), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/PRHS.pdf";js=:remote)


L = create_linear_operator(LaplaceBeltrami(1.0), size(RHSU, 1), size(RHSU, 2)÷2 + 1)
L[1] = 1

LHSU = FastTransforms.sph_zero_spurious_modes!(RHSU./L)
LHSV = sph2fourier(LHSU)
LHSF = zero(LHSV)
A_mul_B!(LHSF, FastTransforms.plan_synthesis(LHSV), LHSV)

LHSF = [mean(LHSF[1,:])*ones(1, size(LHSF, 2)); LHSF; mean(LHSF[end,:])*ones(1, size(LHSF, 2))]
LHSF = [LHSF LHSF[:,1]]

s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = LHSF, cmin = minimum(LHSF), cmax = maximum(LHSF), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/LPLHS.pdf";js=:remote)


NL = create_linear_operator(NonlocalLaplaceBeltrami(1.0, 0.0, 1.5), size(RHSU, 1), size(RHSU, 2)÷2 + 1)
NL[1] = 1

LHSU = FastTransforms.sph_zero_spurious_modes!(RHSU./NL)
LHSV = sph2fourier(LHSU)
LHSF = zero(LHSV)
A_mul_B!(LHSF, FastTransforms.plan_synthesis(LHSV), LHSV)

LHSF = [mean(LHSF[1,:])*ones(1, size(LHSF, 2)); LHSF; mean(LHSF[end,:])*ones(1, size(LHSF, 2))]
LHSF = [LHSF LHSF[:,1]]

s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = LHSF, cmin = minimum(LHSF), cmax = maximum(LHSF), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/NLPLHS.pdf";js=:remote)

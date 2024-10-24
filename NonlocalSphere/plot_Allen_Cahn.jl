using FastTransforms, LinearAlgebra, Makie, Random

include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

Random.seed!(0)

n = 128
U0 = zeros(4n, 8n-1);
U0[1:n, 1:2n-1] = sphrandn(Float64, n, 2n-1)/n
n = size(U0, 1)

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = Float32[cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = Float32[sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = Float32[cospi(θ) for θ in θ, φ in φ]

P = plan_sph2fourier(U0)
PS = plan_sph_synthesis(U0)
F = deepcopy(U0)
Ft = zeros(Float32, size(F, 1)+2, size(F, 2)+1)
lmul!(P, F)
lmul!(PS, F)
fillF!(Ft, F)

scene = Scene(resolution = (1200, 1200));
surf = surface!(scene, x, y, z, color = Ft, colormap = :viridis, colorrange = (-1.0, 1.0));
update_cam!(scene, Vec3f0(2.5), Vec3f0(0), Vec3f0(0, 0, 1))
scene.center = false
scene

T = 8.0
n = round(Int, 6*T)

SLPDE = SemiLinearPDE(LaplaceBeltrami(0.02), NonlinearOperator(u->u-u^3), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io)
Makie.save("plots/LAC.gif", io)
#=
SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.02, 0.0, 0.2), NonlinearOperator(u->u-u^3), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io)
Makie.save("plots/NAC.gif", io)
=#

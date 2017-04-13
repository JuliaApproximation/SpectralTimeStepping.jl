using DiffEqBase, OrdinaryDiffEq, Sundials, ApproxFun
using Base.Test

S=Fourier()
u0=Fun(θ->cos(cos(θ-0.1)),S)
c=Fun(cos,S)

prob = SpectralTimeSteppingProblem((t,u)->u''+(c+1)*u',u0,(0.,1.))
@time u=solve(prob,Tsit5());
@time u=solve(prob,CVODE_BDF());
@time u=solve(prob,ode45());
@time u=solve(prob,radau());

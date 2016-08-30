using ApproxFun, ODE, Plots

import ODE: ode23, ode45, ode78, ode4, ode4ms, ode23s, ode4s

for ode in (:ode23, :ode45, :ode78, :ode4, :ode4ms, :ode23s, :ode4s)
    @eval function $ode(fn, t0, y0, d::Space; reltol=eps(), kwargs...)
        t0added=false
        for logn = 4:20
            t = points(d, 2^logn)
            t0 ∉ t && (unshift!(t,t0); t0added=true)
            t, u = $ode(fn, y0, t; points=:specified, reltol=reltol, kwargs...)
            cf = t0added ? Fun(ApproxFun.transform(d,u[2:end]), d) : Fun(ApproxFun.transform(d,u), d)

            maxabsc=maxabs(cf.coefficients)
            if maxabsc==0 && maxabsfr==0
                return(zeros(d))
            end

            if length(cf) > 8 && maxabs(cf.coefficients[end-8:end]) < reltol
                return chop!(cf,reltol/10)
            end
        end
        warn("Maximum length "*string(2^20+1)*" reached")

        t = points(d, 2^21)
        t0 ∉ t && (unshift!(t,t0); t0added=true)
        t, u = $ode(fn, y0, t; points=:specified, kwargs...)

        t0added ? Fun(ApproxFun.transform(d,u[2:end]), d) : Fun(ApproxFun.transform(d,u), d)
    end
end

sp = Chebyshev(Interval(2,0))

u0 = ode45((t,u)->-u^2/(2-.1^2*u),2.0,-3.0,sp;reltol=eps(Float32),abstol=realmin())

pyplot()

plot(u0)

norm(u0'+u0^2/(2-.1^2*u0))

N=u->[u(2)+3.0;u'+u^2/(2-.1^2*u)]

u = newton(N,u0)

norm(u'+u^2/(2-.1^2*u))
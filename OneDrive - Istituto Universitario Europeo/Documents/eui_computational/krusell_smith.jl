# This is a replication attempt at the Krusell Smith with EGM
# Problem Set 1, Computational Methods
# By Marius Gruenewald

#----------------#
# Importing Pckg #
#----------------#
#import Pkg; Pkg.add("CSV")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("Interpolations")
#import Pkg; Pkg.add("LaTeXStrings")
#import Pkg; Pkg.add("Parameters")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("QuantEcon")
#import Pkg; Pkg.add("BenchmarkTools")
#import Pkg; Pkg.add("StatsBase")
#import Pkg; Pkg.add("TensorCore")
#import Pkg; Pkg.add("GLM")
#import Pkg; Pkg.add("DataFrames")

# using Pckg

using QuantEcon
using BenchmarkTools
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using GLM
using StatsBase
using TensorCore
using LinearAlgebra
using DataFrames
using Plotly

Random.seed!(1234)

@with_kw struct Primitives
    
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 1.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    A::Float64 = 1.0

    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Array{Float64, 2} = [0.9 0.1; 0.1 0.9]
    nz::Int64 = length(z_grid)

    Z_mat::Array{Float64, 2} = [0.5 0.5; 0.1 0.9]
    Z_agg_grid::Vector{Float64} = [0.99, 1.01]
    nZ::Float64 = length(Z_agg_grid)

    minval_agg::Float64 = 5.33*0.9 # 5.33 is the GE capital from previous exercise
    maxval_agg::Float64 = 5.33*1.1
    n_agg::Int64 = 5
    K_agg_grid::Vector{Float64} = exp.(LinRange(log(minval_agg+1),log(maxval_agg+1), n_agg)).-1

    minval::Float64 = 0.0
    maxval::Float64 = 30.0
    na::Int64 = 200
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    tol::Float64 = 1e-5
    maxiter::Int64 = 200
end


function agg_labour(prim::Primitives)
    
    @unpack nz, trans_mat, z_grid = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    tol = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    return Phi_sd, L_Agg[1]
end

function Z_agg_dist(prim::Primitives)
    
    @unpack Z_mat, Z_agg_grid = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    tol = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    return Phi_sd, L_Agg[1]
end

# Rework and solve for 
function firm_decision(prim::Primitives, Agg_L::Float64, K_agg::Float64, Z_agg::Float64)
    
    @unpack α, δ, β = prim

    r = α .* Z_agg * K_agg'.^(α-1).* (Agg_L)^(1-α) .- δ
    wage = (1-α).* Z_agg *(K_agg').^(α) .* (Agg_L)^(-α)
    
   return r, wage
end


function util_(prim::Primitives, c)
    @unpack σ = prim
    u = σ == 1 ? x -> log.(x) : x -> (x.^(1 - σ) .- 1) ./ (1 - σ)
    return u(c)
end


function agg_capital_hh(prim::Primitives, dist_fin::Matrix{Float64}, Kg::Matrix{Float64}, Agg_L::Float64, Z_now::Float64)
    
    @unpack A, α = prim

    agg_temp = sum(dist_fin.* Kg)

    Y = Z_now*(sum(agg_temp))^(α) * (Agg_L)^(1 - α)

    return agg_temp, Y
end



function ALM(K::Float64,Zi::Int64,β_alm::Array{Float64,2}) 
    Kp = exp( β_alm[Zi,:]' * [1., log(K) ] ) 
end 



function sim_Z_shocks(prim::Primitives, n_sim::Int64)

    return simulate(MarkovChain(prim.Z_mat), n_sim),  simulate(MarkovChain(prim.trans_mat), n_sim)
end


# Call this function for transition
function EGM_step(prim::Primitives, cpol::Array{Float64, 4}, V::Array{Float64, 4}, Agg_L::Float64, β_vec::Matrix{Float64})

    # Here, we do not find convergence. We just take the previous results and compute the responses.
    # This gives inputs for next period. We simulate the economy sequentionally.

    @unpack nz, na, nZ, n_agg, z_grid, a_grid_log, Z_agg_grid, K_agg_grid, trans_mat, Z_mat, σ, β = prim
    
    a_grid = copy(a_grid_log)

    V_new = similar(V)
    kpol_new = similar(V)
    cpol_new = similar(V)

    any(cpol .<= 0.) ? cpol = ones(size(cpol)) .* sqrt(eps()) : nothing

    a_endo = zeros(na)
    V_endo = zeros(na)
    r_ = []
    w_ = []
    # Given aggregate state todaz
    for (Z_idx, Z) in enumerate(Z_agg_grid)
        for (K_idx, K) in enumerate(K_agg_grid)

            # Get interest rate and wage for today
            r, w = firm_decision(prim, Agg_L, K, Z)
            # Implied next period aggregate capital
            K_next = ALM(K, Z_idx, β_vec) 

            # Interpolate & Extrapolate to get grid-point-based values
            Vp_itp = extrapolate( interpolate( (a_grid, z_grid, K_agg_grid, Z_agg_grid), V     , Gridded(Linear()) ) , Interpolations.Flat() )
            cp_itp = extrapolate( interpolate( (a_grid, z_grid, K_agg_grid, Z_agg_grid), cpol , Gridded(Linear()) ) , Interpolations.Flat() )

            # for given individual productivity
            for (z_idx, z) in enumerate(z_grid)

                # The decision of today is influenced by both z' and Z' directly via two transition matrices. We need to integrate for both future states.
                # Store for sum 
                EV = zeros(na)
                EU = zeros(na)

                # Start numerical integration
                for (Z_prime_idx, Z_prime) in enumerate(Z_agg_grid)

                    # However, Z' changes r'. Calculate r'
                    r_next, _ = firm_decision(prim, Agg_L, K_next, Z_prime)
                    for (z_prime_idx, z_prime) in enumerate(z_grid)
                        EV += trans_mat[z_idx,z_prime_idx] .* Z_mat[Z_idx,Z_prime_idx] .* Vp_itp(a_grid, z_prime, K_next, Z_prime)
                        EU += (1 + r_next) .* trans_mat[z_idx,z_prime_idx] .* Z_mat[Z_idx,Z_prime_idx] .* cp_itp(a_grid, z_prime, K_next, Z_prime).^(-σ)
                    end
                end
                
                # get Value function and back out current capital given capital_prime
                for (a_next_idx, a_prime) in enumerate(a_grid)
                    # get current consumption
                    c = (β * EU[a_next_idx])^(-1/σ)
                    # back out today's capital
                    a_endo[a_next_idx] = (a_prime + c - z*w)/(1+r)
                    V_endo[a_next_idx] = util_(prim, c) + β * EV[a_next_idx] 

                end

                a_endo_aux = a_endo
                a_grid_aux = a_grid
                V_endo_aux = V_endo 
    
                # interpolate 
                kpol_new[:, z_idx, K_idx, Z_idx] = extrapolate( interpolate( (a_endo_aux,),a_grid_aux,Gridded(Linear())), Interpolations.Flat() )(a_grid) 
                V_new[:, z_idx, K_idx, Z_idx]     = extrapolate( interpolate( (a_endo_aux,),V_endo_aux,Gridded(Linear())), Interpolations.Flat() )(a_grid)
            end

            # back out consumption policy from budget constraint
            Z_ = repeat(reshape(Z_agg_grid, 1, 1, 1, Int(nZ)), na, nz, n_agg, 1)
            r_ = [firm_decision(prim, Agg_L, K_agg_grid[Ki], Z_agg_grid[Zi])[1] for Ki = 1:n_agg, Zi=1:Int(nZ)]
            r_ = repeat(reshape(r_, 1, 1, n_agg, Int(nZ)), na, nz, 1, 1)
            w_ = [firm_decision(prim, Agg_L, K_agg_grid[Ki], Z_agg_grid[Zi])[2] for Ki = 1:n_agg, Zi=1:Int(nZ)]
            w_ = repeat(reshape(w_, 1, 1, n_agg, Int(nZ)), na, nz, 1, 1)

            cpol_new = ( (1 .+ r_ ) .* repeat(a_grid, 1, nz, n_agg, Int(nZ)) .+ repeat(z_grid', na, 1, n_agg, Int(nZ)) .* w_ .- kpol_new )

        end
    end

    return V_new, cpol_new, kpol_new
end


# EGM including EGM_step
function EGM(prim::Primitives, β_vec::Matrix{Float64}, cpol::Array{Float64, 4}, V::Array{Float64, 4}, Agg_L::Float64)

    @unpack maxiter, tol, na, nz, nZ, n_agg = prim

    any(cpol .<= 0.) ? cpol = ones(size(cpol)) .* sqrt(eps()) : nothing
    k_pol = [] 

    # find fixed point
    ϵ = 1e-10
    dist = 1 + ϵ
    max_iter = 10_000
    ite = 0
    while (dist > ϵ) && (ite  < max_iter)
        ite += 1
        (ite % 25 == 0) ? println(ite," ",dist) : nothing
   
        V_new, c_pol_new, k_pol  = EGM_step(prim, cpol, V, Agg_L, β_vec) 
 
        # update distance 
        dist    = norm( V .- V_new) / (1 + norm(V) )
        V       = deepcopy(V_new)
        cpol   = deepcopy(c_pol_new)

        if ite == max_iter
            @warn "NOT CONVERGED AFTER MAX ITERATIONS"
        end
        
    end # while
  

    return V, cpol, k_pol
end




function K_state(Z_seq::Vector{Int64}, β₀ₗ::Float64, β₁ₗ::Float64, β₀ₕ::Float64, β₁ₕ::Float64, n_sim::Int64)

    mc = QuantEcon.tauchen(nz,ρ,σ_e,μ,2)
    z_logs = mc.state_values
    trans_mat = mc.p

    # Initial guess 1
    K_seq = ones(n_sim+1)

    for (i,Z) in enumerate(Z_seq)
        if Z == 1
            K_seq[i+1] = β₀ₗ + β₁ₗ*K_seq[i]
        else 
            K_seq[i+1] = β₀ₕ + β₁ₕ*K_seq[i]
        end
    end


    return K_seq
end


function simulate_JD(prim::Primitives, k_pol::Array{Float64, 4}, Z_index::Vector{Int64}, n_sim::Int64, JD_init::Any, Agg_L::Float64)
  
    @unpack a_grid_log, z_grid, minval, maxval, K_agg_grid, Z_agg_grid, trans_mat, Z_mat, na, nz, nZ, n_agg = prim 
 
    # simulation grid (finer)
    a_grid = copy(a_grid_log)
    factor = 2
    JD_init = []
    # Since we use log_grid
    k_grid_sim = exp.( range( log(1+minval), log(1+maxval), factor * na)  ) .-1 
    
    # If linear grid
    #k_grid_sim = k_grid
    nks = length(k_grid_sim)
    JD_ts = zeros(nks,nz,n_sim)
    # setup distribution
    ki = searchsortedfirst(k_grid_sim, mean(K_agg_grid))
    # If no guess provided, make a guess
    if JD_init == []
        pdf_z = [1- Agg_L,  Agg_L]  
        JD_ts[ki,:,1] .= pdf_z
        #JD_ts[:,:,1] .= 1. / (nz*nks)
    else
        # If we have a guess
        JD_ts[:,:,1]  .= JD_init
    end
    K_ts  = zeros(n_sim)
    # policy interpolant
    k_pol_itp = extrapolate( interpolate( (a_grid, z_grid, K_agg_grid, Z_agg_grid), k_pol, Gridded(Linear()) ) , Interpolations.Flat() )
  
    # simulate forward
    for t = 1:(n_sim-1)

        (t%1000==0) ? println( t,"/", n_sim ) : nothing

        # aggregate capital
        K_ts[t] = sum( JD_ts[:,:,t]' * k_grid_sim )
 
        for ksi = 1:nks, zi = 1:nz
              
            # find k' policy lower index and associated weight
            Zi      = Z_index[t] 
            kps     = k_pol_itp(k_grid_sim[ksi], z_grid[zi], K_ts[t], Z_agg_grid[Zi])              # policy kp
            kpsi    = minimum([nks-1, searchsortedlast(k_grid_sim, kps ) ])                 # lower index
            kpsi    = maximum([kpsi,1])                                                     # in case idx == 0 
            dist_l  = (kps - k_grid_sim[kpsi]) / (k_grid_sim[kpsi+1] - k_grid_sim[kpsi])    # relative distance to lower point
            wgt     = (1-dist_l)                                                            # weigth for lower index
 
            for zpi = 1:nz 
                # Lower point & weight
                JD_ts[kpsi    , zpi, t+1]  += trans_mat[zi,zpi]  *    wgt   * JD_ts[ksi,zi,t]
                # Upper point & weights
                JD_ts[kpsi + 1, zpi, t+1]  += trans_mat[zi,zpi]  * (1-wgt ) * JD_ts[ksi,zi,t]
            end

        end # ks, z
     
    end # t
    
    # aggregate capital
    K_ts[n_sim] = sum( JD_ts[:,:, n_sim]' * k_grid_sim )

    return JD_ts, k_grid_sim, K_ts 

end


function krusell_smith(n_sim)

    prim = Primitives()

    # Initial Guess of distribution
    #dist = (1/(prim.nz * prim.na)) * ones(prim.nz, prim.na, n_sim)
    V = ones(prim.na, prim.nz, prim.n_agg, Int(prim.nZ))
    cpol = ones(prim.na, prim.nz, prim.n_agg, Int(prim.nZ))
    kpol = ones(prim.na, prim.nz, prim.n_agg, Int(prim.nZ))

    # Guess ALM
    β_vec = [0.3 0.7;
     0.2 0.8]

    #simulation container 
    dist_alm_ = []
    dist, k_sim, K_sim  = [],[],[],[]
    dist_init = []

    # Start
    _, Agg_L = agg_labour(prim)
    Z_index, _ = sim_Z_shocks(prim, n_sim)
    
    Plots.plot(prim.Z_agg_grid[Z_index[1:100]],title="Aggregate Productivity",xlabel="periods",ylabel="Aggregate Productivity",legend=:false)

    count = 0
    dist_alm = 1.0

    while (dist_alm > prim.tol) && (count < prim.maxiter)

        count += 1
        # Compute Aggregate Capital Supply by Households
        V, cpol, kpol = EGM(prim, β_vec, cpol, V, Agg_L)

        # Simulation
        dist, k_sim, K_sim = simulate_JD(prim, kpol, Z_index, n_sim, dist_init, Agg_L)
        # Make new starting guess
        dist_init = dist[:,:,end]

        Y  = log.(K_sim[2:end])
        X  = hcat(ones(n_sim-1), log.(K_sim[1:end-1] )) # one less time because of lag, omit last period

        # Omit first 500 periods and last 
        Y1 = Y[Z_index[1:end-1] .== 1][(500+1):end]       # if bad  aggregate state
        X1 = X[Z_index[1:end-1] .== 1,:][(500+1):end,:]     # if bad  aggregate state
        Y2 = Y[Z_index[1:end-1] .== 2][(500+1):end]       # if good aggregate state
        X2 = X[Z_index[1:end-1] .== 2,:][(500+1):end,:]     # if good aggregate state

        coeff = vcat( (inv(X1'X1)*X1'Y1)' , (inv(X2'X2)*X2'Y2)' )

        # update β_alm
        χ = 0.2 # dampening
        β_new   = χ .* coeff + (1-χ) .* β_vec
        dist_alm  = maximum(abs.( β_vec .- β_new ))
        β_vec  = deepcopy(β_new)
        push!(dist_alm_, dist_alm)

        # output 
        println(count," convergence ALM: ",round(dist_alm_[end],digits=11) ,"   ",round.(β_vec,digits=4) )  
        
    end

    return V, cpol, kpol, dist, k_sim, K_sim, Z_index, β_vec, dist_alm_, count
end


V, cpol, kpol, dist, k_sim, K_sim, Z_index, β_vec, dist_alm, count = krusell_smith(10000)

prim = Primitives()

# plot convergence
P1 =  Plots.plot( title="Convergence ALM", ylabel="log distance" , xlabel="update steps")  ;
P1 =  plot!(1:count, log.(dist_alm), c=:green, label="lom update difference", xticks=1:count )  ;
P1 =  hline!( log.([prim.tol]), c=:red, ls=:dash, label="tolerance"   )  ; 
display( Plots.plot(  P1, layout = (1,1)) ) 

println(" Aggregate Law of Motion: " )
display(β_vec)

_, Agg_L = agg_labour(prim)
function rate(K::Float64, Z::Float64, prim::Primitives, Agg_L::Float64)
    @unpack α, δ = prim
    return  Z * α * (Agg_L / K) ^ (1 - α) - δ
end

Z_sim = prim.Z_agg_grid[Z_index]
r_sim =  [ rate(K_sim[t], Z_sim[t], prim, Agg_L)  for t = 1:10_000 ]

# Time Series plot convergence
P1 = Plots.plot(title="Capital", xlabel="periods");
plot!(1:10_000, K_sim, label = "")  
hline!( [mean(K_sim)] , ls = :dash,c = :red, label="mean") 
hline!( prim.K_agg_grid ,ls=:dash,c=:grey, label = "capital grid points")
P2 = Plots.plot(title="Rate", xlabel="periods");
plot!(1:10_000, r_sim , label = "")  
hline!( [mean(r_sim)] , ls = :dash,c = :red, label="")  
Plots.plot(P1,P2,layout=(2,1),size=(750,750),left_margin=10Plots.mm,bottom_margin=10Plots.mm,legend=:outertopright)

# R² 
df = DataFrame( :kp => log.(K_sim[(500+1):end]),
                :k =>  log.(K_sim[500:end-1]),
                :z =>  log.(Z_sim[500:end-1]),
                :Zi =>   Z_index[500:end-1] )  
df1 = df[df.Zi.==1,:]                
df2 = df[df.Zi.==2,:]                
fit1 = lm( @formula(kp ~ k ),df1,)
fit2 = lm( @formula(kp ~ k ),df2,)
println("R² of ALM(1): ",round(r2(fit1),digits=8) )
println("R² of ALM(2): ",round(r2(fit2),digits=8) )

#### Den Haan

# pick 100 periods in the time series
ts_ = 1000:1100
TT = length(ts_)

# simulate from ALM
K_ts_ALM = ones(TT) * K_sim[ts_[1]]
for tt in 1:TT-1
    t = ts_[tt]
    K_ts_ALM[tt+1] = ALM(K_ts_ALM[tt], Z_index[t], β_vec)
end

error = abs.(K_sim[ts_] ./ K_ts_ALM .- 1.)*100
println("Average (max) Den Haan error over $TT periods: ",round(mean(error),digits=4),"% (",round(maximum(error),digits=4) ,"%)")

# Plot K based on policy vs K based on ALM
Plots.plot(size=(750,400),title="Den Haan Measure",legend=:outertopright)
plot!(ts_, K_sim[ts_],label="Agent policy rule")
plot!(ts_,K_ts_ALM, label="ALM")


##### Distributions
# plot Aggregate distribution
 
P1 = Plots.plot(title="Aggregate Capital" );
histogram!(K_sim , label = "")  
vline!( [mean(K_sim)] ,lw = 2.5, ls = :dash,c = :red, label="mean")

P2 = Plots.plot(title="Rate" );
histogram!(r_sim , label = "") 
vline!( [mean(r_sim)]  ,lw = 2.5, ls = :dash,c = :red, label="")

P3 = Plots.plot(title="Productivity" );
histogram!(1:10_000, Z_sim , label = "") 
vline!( [mean(Z_sim)] ,lw = 2.5, ls = :dash,c = :red, label="")
  
Plots.plot(P1,P2,P3,layout=(3,1),size=(750,750),left_margin=10Plots.mm,bottom_margin=10Plots.mm,legend=:outertopright)
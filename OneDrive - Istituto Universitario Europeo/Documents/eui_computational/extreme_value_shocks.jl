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
#import Pkg; Pkg.add("StatsFuns")

# using Pckg

using BenchmarkTools
using Plots
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore
using StatsFuns


Random.seed!(1234)

@with_kw struct discrete_params_evs
    
    β::Float64 = 0.96
    σ::Float64 = 1.0
    Φ::Float64 = 0.75
    sigma_ϵ::Float64 = 0.01

    n::Float64 = 1.0

    minval::Float64 = 1e-10
    maxval::Float64 = 10.0
    na::Int64 = 1000
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    z_grid::Vector{Float64} = [1.0]
    trans_mat_z::Matrix{Float64} = ones(1,1)
    nz::Int64 = length(z_grid)

    tol::Float64 = 1e-8
    maxiter::Int64 = 10_000
end

function util_(prim::discrete_params_evs, c)
    u = prim.σ == 1 ? x -> log.(x) : x -> (x.^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
    return u(c)
end

function value_function_step(prim::discrete_params_evs, V::Matrix{Float64}, r::Float64, wage::Float64)

    @unpack a_grid_log, z_grid, β, trans_mat_z, na, nz, Φ, n = prim

    # choice-specific value, 1 = working, 2 not working
    V_temp = ones(na, nz, na, 2) * -Inf 

    for (z_idx, z) in enumerate(z_grid)
        
        EV = V * trans_mat_z[z_idx, :]

        for (a_idx, a) in enumerate(a_grid_log)

            for (a_prime_idx, a_prime) in enumerate(a_grid_log)

                # Working
                c_1 = n * wage * z + (1+r)*a - a_prime
                # Not working
                c_0 = (1+r)*a - a_prime

                if (c_0 <= 0.0) && (c_1 <= 0.0)
                    V_temp[a_idx, z_idx, a_prime_idx, :] .= -Inf
                    break # shortcut due to monotonicity
                elseif (c_1 <= 0.0)
                    V_temp[a_idx, z_idx, a_prime_idx, 1] = util_(prim, c_0) + β*EV[a_prime_idx]
                    V_temp[a_idx, z_idx, a_prime_idx, 2] = -Inf
                elseif (c_0 <= 0.0)
                    V_temp[a_idx, z_idx, a_prime_idx, 1] = -Inf
                    V_temp[a_idx, z_idx, a_prime_idx, 2] = util_(prim, c_1) - Φ * n + β*EV[a_prime_idx]
                else
                    V_temp[a_idx, z_idx, a_prime_idx, 1] = util_(prim, c_0) + β*EV[a_prime_idx]
                    V_temp[a_idx, z_idx, a_prime_idx, 2] = util_(prim, c_1) - Φ * n + β*EV[a_prime_idx]
        
                end

            end
        
        end
    
    end

    # maximize value function and update distance
    V_new = maximum(V_temp, dims= (3,4) )[:,:,1,1]  # maximizing over both choices
    V_d_new = maximum(V_temp, dims= (3) )[:,:,1,:]  # discrete choice specific

    return V_new, V_d_new, V_temp
end

function value_function_iter(prim::discrete_params_evs, r::Float64, wage::Float64)

    @unpack a_grid_log, z_grid, β, trans_mat_z, na, nz, Φ, n, tol, maxiter = prim

    # Storage Space
    V = ones(na, nz)

    k_pol = zeros(na, nz)
    c_pol = zeros(na, nz)
    n_pol = zeros(na, nz)

    # choice-specific value
    V_temp = []

    err = 1
    count = 1

    while err > tol && count < maxiter

        count += 1
        (count%100 == 0) ? println(count," ", err) : nothing

        V_new, _, V_temp = value_function_step(prim, V, r, wage)

        err = maximum(abs.(V_new - V))
        V = copy(V_new)
        
    end
    k_pol_idx = argmax(V_temp, dims= (3,4) )[:,:,1,1]

    for a_idx in 1:na
        for z_idx in 1:nz
            k_pol[a_idx, z_idx] = a_grid_log[k_pol_idx[a_idx, z_idx][3]]
            n_pol[a_idx, z_idx] = [0, 1][k_pol_idx[a_idx, z_idx][4]]
        end 
    end
    c_pol = (1+r)*a_grid_log  .+  n_pol .* z_grid'*wage .-  k_pol

    return V, c_pol, k_pol, n_pol

end


function solve_model()

    prim = discrete_params_evs()
    r = 0.03
    wage = 1.0

    V, c_pol, k_pol, n_pol = value_function_iter(prim, r, wage)

    return V, c_pol, k_pol, n_pol

end

V,c_pol,k_pol,n_pol = solve_model()

prim = discrete_params_evs()
a_grid = prim.a_grid_log
plotly(size=(700,450),lw=2.5) 

P1 = plot(title="Value Function",xlabel="assets",ylabel="value"); 
plot!(a_grid, V, label="w/o EV",legend=false)          
vline!(a_grid[findall(  [0; diff(n_pol,dims=1)] .!= 0.0 )] ,c=:red,ls=:dot,label="discrete choice",legend=false)    
  
P2 = plot(title="Asset Policy",xlabel="assets",ylabel="asset choice");
plot!(a_grid, k_pol,label="",legend=false)       
plot!(a_grid, a_grid, ls=:dot, c=:grey, label="45°")
vline!(a_grid[findall(  [0;diff(n_pol,dims=1)] .!= 0.0 )], c=:red, ls=:dot, label="", legend=false)      

P3 = plot(title="Consumption Policy",xlabel="assets",ylabel="consumption choice");
plot!(a_grid, c_pol,label="",legend=false)   
vline!(a_grid[findall(  [0;diff(n_pol,dims=1)] .!= 0.0 )], c=:red,ls=:dot,label="", legend=false)   
 
P4 = plot(title="Discrete Policy",xlabel="assets",ylabel="probability working");
plot!(a_grid, n_pol,label="",legend=false)     

plot(P1,P2,P3,P4,size=(900,700),bottom_margin=10Plots.mm,legend=:outertopright)


###################################################################################################
################################################ EVS ##############################################


function evs_iter(prim::discrete_params_evs, r::Float64, wage::Float64)

    @unpack a_grid_log, z_grid, β, trans_mat_z, na, nz, Φ, n, tol, maxiter, sigma_ϵ = prim

    # Storage Space
    V_1 = ones(na, nz)
    V_2   = zeros(na, nz)

    k_pol_1 = zeros(na, nz, 2)
    P_pol = zeros(na, nz, 2)

    # choice-specific value
    V_temp, V_3 = [], []

    err = 1
    count = 1

    while err > tol && count < maxiter

        count += 1
        (count%100 == 0) ? println(count," ", err) : nothing

        _, V_3, V_temp = value_function_step(prim, V_1, r, wage)

        for a_idx = 1:na, z_idx = 1:nz
            P_pol[a_idx, z_idx,:] = StatsFuns.softmax( V_3[a_idx, z_idx,:] ./ sigma_ϵ )  
            V_2[a_idx, z_idx]   = sigma_ϵ * StatsFuns.logsumexp( V_3[a_idx, z_idx,:] ./ sigma_ϵ )  
        end

        err = maximum(abs.(V_2 - V_1))
        V_1 = deepcopy(V_2)
        
    end

    k_pol_idx = argmax(V_temp, dims= (3) )[:,:,1,:]

    for a_idx in 1:na
        for z_idx in 1:nz
            for d_idx in 1:2
                k_pol_1[a_idx, z_idx, d_idx] = a_grid_log[k_pol_idx[a_idx, z_idx, d_idx][3]]
            end
        end 
    end
    d_ten = cat(zeros(na,nz,1), ones(na,nz,1), dims=3)
    c_pol_1 = (1+r)* repeat(a_grid_log,1,nz,2)  .+  d_ten .* repeat(z_grid',na,1,2)*wage   .-  k_pol_1

    return V_1, V_3, c_pol_1, k_pol_1, P_pol

end

function solve_evs()
    prim = discrete_params_evs()
    r = 0.03
    wage = 1.0

    V_1, V_2, c_pol_1, k_pol_1, P_pol = evs_iter(prim, r, wage)

    return V_1, V_2, c_pol_1, k_pol_1, P_pol
end

V_1, V_2, c_pol_1, k_pol_1, P_pol = solve_evs()

plot!(P1, prim.a_grid_log, V_1, label="w/ EV", legend=:topleft, c=:magenta)      
plot!(P2, prim.a_grid_log ,sum(k_pol_1 .* P_pol,dims=3)[:,:,1], label="", legend=:topleft,c=:magenta)      
plot!(P3, prim.a_grid_log, sum(c_pol_1 .* P_pol,dims=3)[:,:,1], label="", legend=:topleft,c=:magenta)  
plot!(P4, prim.a_grid_log, P_pol[:,:,2],label="", legend=:topright, c=:magenta)    

plot(P1,P2,P3,P4, size=(900,700), bottom_margin=10Plots.mm, legend=:outertopright)
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
using LinearAlgebra
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore
using StatsFuns


Random.seed!(1234)

function parameter_toy_model()
    
    minval::Float64 = 1e-10
    maxval::Float64 = 25.0
    na::Int64 = 100
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    #a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    z_grid::Vector{Float64} = [1.0]
    trans_mat_z::Matrix{Float64} = ones(1,1)
    nz::Int64 = length(z_grid)

    h_rent_min = 1e-10
    h_rent_max = 5
    n_rent = 20
    h_rent_grid = collect(range(h_rent_min, h_rent_max, n_rent))

    h_own_min = 1e-10
    h_own_max = 10
    n_own = 20
    h_own_grid = collect(range(h_own_min, h_own_max, n_own))

    p_own = 5
    p_rent = p_own/20

    pm = (

        β = 0.96,
        σ = 2.0,
        ν = 0.7,
        δ_h = 0.02,
        Φ = 0.1,

        w = 1,
        r = 0.03,

        p_own = p_own,
        p_rent = p_rent,
        M = 30,
        R = 0.01,

        amin = minval,
        amax = maxval,
        na = na,
        a_grid = a_grid_lin,

        nz = nz,
        z_grid = z_grid,
        Pr_z = trans_mat_z,

        hr_min = h_rent_min,
        hr_max = h_rent_max,
        n_rent = n_rent,
        h_rent_grid = h_rent_grid,

        ho_min = h_own_min,
        ho_max = h_own_max,
        n_own = n_own,
        h_own_grid = h_own_grid,

    )
end


function util_(prim, c, h, discount)

    @unpack σ, ν, δ_h = prim

    if discount == false
        return ((c.^prim.ν * h.^(1 - prim.ν)).^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
    else
        return ((c.^prim.ν * ((1 - δ_h) * h.^(1 - prim.ν))).^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
    end
end

function mortgage(prim, h_own_grid)
    return (1 - prim.Φ) * prim.p_own * h_own_grid
end


function mortgage_payments(prim, size)

    @unpack M, R = prim
    compunded_interest = 0
    for k in 1:M
        compunded_interest += 1/((1+R)^k)
    end

    return size./compunded_interest

end


function value_function_step(prim, V)

    @unpack a_grid, z_grid, h_rent_grid, h_own_grid, β, Pr_z, na, nz, n_own, n_rent, δ_h, p_own, p_rent, r, w, Φ = prim
    
    V = ones(na, nz)
    # choice-specific value, 1 = renting, 2 buying, 3 owning
    V_rent = ones(na, nz, na, n_rent, 3) * -Inf
    b_prime_0 = mortgage(prim, h_own_grid)
    mortgage_payment = mortgage_payments(prim, b_prime_0)

    iter = 0
    tol = 0.0001
    err = 1.0
    maxiter = 500

    while (err > tol) && (iter < maxiter)

        iter += 1
        println(iter," ", err)

        for (z_idx, z) in enumerate(z_grid)
            
            EV = V * Pr_z[z_idx, :]

            for (r_idx, rent_now) in enumerate(h_rent_grid)

                for (b_prime_idx, buy_prime) in enumerate(h_own_grid)

                    #b_prime = (1 - Φ) * p_own * buy_prime

                    for (a_idx, a) in enumerate(a_grid)

                        for (a_prime_idx, a_prime) in enumerate(a_grid)

                            c_rent = w*z + (1+r)*a - a_prime - p_rent*rent_now # renting
                            c_buy = w*z + (1+r)*a - a_prime - p_rent*rent_now + b_prime_0[b_prime_idx] - Φ * p_own * buy_prime - mortgage_payment[b_prime_idx] #buying
                            # add owning

                            if (c_rent <= 0.0) & (c_buy <= 0.0) 
                                break # shortcut due to monotonicity

                            elseif (c_rent <= 0.0)  & (c_buy > 0.0)
                                V_rent[a_idx, z_idx, a_prime_idx, b_prime_idx, 2] = util_(prim, c_buy, buy_prime, false) + β*EV[a_prime_idx]   

                            elseif (c_rent >= 0.0)  & (c_buy < 0.0)
                                V_rent[a_idx, z_idx, a_prime_idx, r_idx, 1] = util_(prim, c_rent, rent_now, false) + β*EV[a_prime_idx]

                            else
                                V_rent[a_idx, z_idx, a_prime_idx, r_idx, 1] = util_(prim, c_rent, rent_now, false) + β*EV[a_prime_idx] 
                                V_rent[a_idx, z_idx, a_prime_idx, b_prime_idx, 2] = util_(prim, c_buy, buy_prime, false) + β*EV[a_prime_idx]                
              
                            end

                        end
                    
                    end

                end

            end
        
        end
        V_new = maximum(V_rent, dims= (3,4,5) )[:,:,1,1]

        err  = norm( V .- V_new) / (1 + norm(V) )
        V    = deepcopy(V_new)

    end
    # maximize value function and update distance
    # maximizing over both choices
    V_k_new = maximum(V_rent, dims= (3) )[:,:,:,1] # assets choice specific
    V_h_new = maximum(V_rent, dims= (4) )[:,:,1,:] # discrete choice specific

    return V_new, V_d_new, V_temp
end

V_new = maximum(V_rent, dims= (3,4,5) )[:,:,1,1] 

plot(V_new) # Value Function
plot(maximum(V_rent[:,:,:,:,1], dims = (3,4))[:,:,1,1])

kpi_di_idx = argmax(V_rent[:,:,:,:,1],dims= (3,4,5) )[:,:,1,1]
k_pol   = [a_grid[kpi_di_idx[ki,zi][3]] for ki = 1:na, zi=1:nz]
h_pol   = [h_rent_grid[kpi_di_idx[ki,zi][4]] for ki = 1:na, zi=1:nz] 
d_pol = [[0,1][kpi_di_idx[ki,zi][5]] for ki = 1:na, zi=1:nz] 

c_pol   = (1+r).*a_grid.*ones(na, n_rent) - p_rent.* h_pol' .*ones(na, n_rent) .* z_grid'*w   .-  k_pol.*ones(na, n_rent)

plot(k_pol)
plot(h_pol)

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

    prim = parameter_toy_model()
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
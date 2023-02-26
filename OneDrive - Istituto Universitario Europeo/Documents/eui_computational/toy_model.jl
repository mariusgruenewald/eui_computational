# Toy Model for the second year forum
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
using Random
using LinearAlgebra
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore
using StatsFuns
using Interpolations
using Random

Random.seed!(1234)

function parameter_toy_model()
    
    minval::Float64 = 1e-10
    maxval::Float64 = 25.0
    na::Int64 = 100
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    #a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Array{Float64, 2} = [0.9 0.1; 0.1 0.9]
    nz::Int64 = length(z_grid)

    h_rent_min = 1e-10
    h_rent_max = 10
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
        ν = 0.5,
        δ_h = 0.1,
        Φ = 0.1,
        s = 0.98,

        w = 1,
        r = 0.03,
        J = 70,

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
        Pr_z = trans_mat,

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
        return ((c.^prim.ν * (h + 0.0001).^(1 - prim.ν)).^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
    else
        return ((c.^prim.ν * ((1 - δ_h) * (h + 0.0001).^(1 - prim.ν))).^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
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

function utility_loop(prim)
    @unpack a_grid, z_grid, h_own_grid, n_own, δ_h, p_own, r, w, J, na, nz = prim

    cons_own = zeros(J, na, na, nz, n_own, n_own)
    util_own = zeros(J, na, na, nz, n_own, n_own)

    for j in 1:J
        for (k_i, k) in enumerate(a_grid)
            for (kp_i, kp) in enumerate(a_grid)
                for (z_i, z) in enumerate(z_grid)
                    for (h_i, h) in enumerate(h_own_grid)
                        for (hp_i, hp) in enumerate(h_own_grid)
                            c_now = w*z + (1+r)*k - kp - p_own*hp + (1-δ_h)*p_own*h
                            if (c_now >= 0.0)
                                cons_own[j, k_i, kp_i, z_i, h_i, hp_i] = c_now
                                util_own[j, k_i, kp_i, z_i, h_i, hp_i] = util_(prim, c_now, h, true)
                            end
                        end
                    end
                end
            end
        end
        println("Utility of all combinations at age ", j)

    end

    return cons_own, util_own
end



function value_function(prim, util_own)

    @unpack a_grid, z_grid, h_own_grid, β, Pr_z, na, nz, n_own, δ_h, p_own, r, w, J, s = prim
    
    V = zeros(J, na, n_own, nz)
    V_next = zeros(na, n_own, nz)
    # choice-specific value, 1 = renting, 2 buying
    V_temp = []
    #b_prime_0 = mortgage(prim, h_own_grid)
    #mortgage_payment = mortgage_payments(prim, b_prime_0)

    pol_a = zeros(J, na, n_own, nz)
    pol_own = zeros(J, na, n_own, nz)
    pol_c = zeros(J, na, n_own, nz)

    for j in reverse(1:J-1)

        for (z_idx, z) in enumerate(z_grid)
        
            for (a_idx, a_now) in enumerate(a_grid)

                for (o_idx, own_now) in enumerate(h_own_grid)

                    V_temp = util_own[j, a_idx, :, z_idx, o_idx, :] .+  β * s .* V[j+1, :, :, :] ⊡ Pr_z[z_idx, :]
                    V_next[a_idx, o_idx, z_idx] = maximum(V_temp)
                    Opt_ind = argmax(V_temp)
                    pol_a[j, a_idx, o_idx, z_idx] = Opt_ind[1]
                    pol_own[j, a_idx, o_idx, z_idx] = Opt_ind[2]
                    pol_c[j, a_idx, o_idx, z_idx] = w*z + (1+r)*a_now - Opt_ind[1] - p_own*Opt_ind[2] + (1-δ_h)*p_own*own_now
                
                end
            
            end
        
        end

        V[j, :, :, :] = V_next
        println("Done with age ", j)

    end

    return pol_a, pol_own, pol_c, V
end

function inheritance(prim)

    @unpack h_own_grid = prim

    new_grid = h_own_grid .+ 1

    

    return 
end


function value_function_inheritance(prim, util_own)

    @unpack a_grid, z_grid, h_own_grid, β, Pr_z, na, nz, n_own, δ_h, p_own, r, w, J, s = prim
    
    V = zeros(J, na, n_own, nz)
    V_next = zeros(na, n_own, nz)
    # choice-specific value, 1 = renting, 2 buying
    V_temp = []
    #b_prime_0 = mortgage(prim, h_own_grid)
    #mortgage_payment = mortgage_payments(prim, b_prime_0)

    pol_a = zeros(J, na, n_own, nz)
    pol_own = zeros(J, na, n_own, nz)
    pol_c = zeros(J, na, n_own, nz)

    inheritance = 1
    o_idx_new = 0
    own_new = 0

    for j in reverse(1:J-1)

        for (z_idx, z) in enumerate(z_grid)
        
            for (a_idx, a_now) in enumerate(a_grid)

                for (o_idx, own_now) in enumerate(h_own_grid)

                    if j == 50
                        own_new = own_now + inheritance
                        o_idx_new = argmin(abs.(h_own_grid .- own_new))

                        V_temp = util_own[j, a_idx, :, z_idx, o_idx_new, :] .+  β * s .* V[j+1, :, :, :] ⊡ Pr_z[z_idx, :]
                        V_next[a_idx, o_idx, z_idx] = maximum(V_temp)
                        Opt_ind = argmax(V_temp)
                        pol_a[j, a_idx, o_idx, z_idx] = Opt_ind[1]
                        pol_own[j, a_idx, o_idx, z_idx] = Opt_ind[2]
                        pol_c[j, a_idx, o_idx, z_idx] = w*z + (1+r)*a_now - Opt_ind[1] - p_own*Opt_ind[2] + (1-δ_h)*p_own*own_new

                    else

                        V_temp = util_own[j, a_idx, :, z_idx, o_idx, :] .+  β * s .* V[j+1, :, :, :] ⊡ Pr_z[z_idx, :]
                        V_next[a_idx, o_idx, z_idx] = maximum(V_temp)
                        Opt_ind = argmax(V_temp)
                        pol_a[j, a_idx, o_idx, z_idx] = Opt_ind[1]
                        pol_own[j, a_idx, o_idx, z_idx] = Opt_ind[2]
                        pol_c[j, a_idx, o_idx, z_idx] = w*z + (1+r)*a_now - Opt_ind[1] - p_own*Opt_ind[2] + (1-δ_h)*p_own*own_now
                    end
                
                end
            
            end
        
        end

        V[j, :, :, :] = V_next
        println("Done with age ", j)

    end

    return pol_a, pol_own, pol_c, V
end





function solve_model()

    prim = parameter_toy_model()

    cons_own, util_own = utility_loop(prim)

    pol_a, pol_own, pol_c, V = value_function(prim, util_own)
    pol_a_i, pol_own_i, pol_c_i, V_i = value_function_inheritance(prim, util_own)

    return cons_own, util_own, pol_a, pol_own, pol_c, V, pol_a_i, pol_own_i, pol_c_i, V_i

end

cons_own, util_own, pol_a, pol_own, pol_c, V, pol_a_i, pol_own_i, pol_c_i, V_i = solve_model();

prim = parameter_toy_model();

# Compare 
Plots.plot(a_grid, [a_grid[Int.(pol_a[20, :, 10, 1])] a_grid[Int.(pol_a_i[20, :, 10, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[45, 80, :, 2])] h_own_grid[Int.(pol_own_i[45, 80, :, 2])]])


Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[10, :, :, 1]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")

Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[10, :, :, 2]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")



######################## Surface Plot of Value function
## Age 20
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[20, :, :, 1]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[20, :, :, 2]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")

## Age 40
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[40, :, :, 1]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[40, :, :, 2]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")

## Age 60
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[60, :, :, 1]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), V[60, :, :, 2]')
xlabel!("Asset Grid")
ylabel!("Housing Grid")
title!("Value Function at Grid Points")



############################ Plotting Policy Functions ##########################################
############### Capital

mesh_grid = a_grid * h_own_grid'

## Age 20
Plots.surface(range(prim.amin, prim.amax, prim.na), range(prim.ho_min, prim.ho_max, prim.n_own), mesh_grid[Int.(pol_a[20, :, :, 2])]')
Plots.plot(a_grid, [a_grid[Int.(pol_a[20, :, 10, 1])] a_grid[Int.(pol_a[20, :, 10, 2])]])
Plots.plot(a_grid, [a_grid[Int.(pol_a[20, :, 19, 1])] a_grid[Int.(pol_a[20, :, 19, 2])]])

# Age 40
Plots.plot(a_grid, [a_grid[Int.(pol_a[40, :, 2, 1])] a_grid[Int.(pol_a[40, :, 2, 2])]])
Plots.plot(a_grid, [a_grid[Int.(pol_a[40, :, 10, 1])] a_grid[Int.(pol_a[40, :, 10, 2])]])
Plots.plot(a_grid, [a_grid[Int.(pol_a[40, :, 19, 1])] a_grid[Int.(pol_a[40, :, 19, 2])]])

# Age 60
Plots.plot(a_grid, [a_grid[Int.(pol_a[60, :, 2, 1])] a_grid[Int.(pol_a[60, :, 2, 2])]])
Plots.plot(a_grid, [a_grid[Int.(pol_a[60, :, 10, 1])] a_grid[Int.(pol_a[60, :, 10, 2])]])
Plots.plot(a_grid, [a_grid[Int.(pol_a[60, :, 19, 1])] a_grid[Int.(pol_a[60, :, 19, 2])]])


############### Housing
## Age 20
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[20, 2, :, 1])] h_own_grid[Int.(pol_own[20, 2, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[20, 10, :, 1])] h_own_grid[Int.(pol_own[20, 10, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[20, 19, :, 1])] h_own_grid[Int.(pol_own[20, 19, :, 2])]])

# Age 40
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[40, 2, :, 1])] h_own_grid[Int.(pol_own[40, 2, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[40, 10, :, 1])] h_own_grid[Int.(pol_own[40, 10, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[40, 19, :, 1])] h_own_grid[Int.(pol_own[40, 19, :, 2])]])

# Age 60
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[60, 2, :, 1])] h_own_grid[Int.(pol_own[60, 2, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[60, 10, :, 1])] h_own_grid[Int.(pol_own[60, 10, :, 2])]])
Plots.plot(h_own_grid, [h_own_grid[Int.(pol_own[60, 19, :, 1])] h_own_grid[Int.(pol_own[60, 19, :, 2])]])


########################### Aggregate Outcomes ##################

N = 1000


s_rand = rand(Random.seed!(1234),N, prim.J)

# create identifier where they die
death_age = zeros(N)
death_age[:] .= 71
for sim = 1:N
    for age = 2:J
        if s_rand[sim,age] > prim.s
            death_age[sim] = age
            break
        end
    end
end



# Compute Invariant Distribution
function invariant_prod_dist(prim)

    @unpack nz, z_grid, Pr_z = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    tol::Float64 = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*Pr_z
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    z_grid = z_grid./(z_grid' *Phi_sd[1,:])

    return Phi_sd[1,:], z_grid
end

prod_dist, _ = invariant_prod_dist(prim)

# Simulate sequence of shocks
z_path_index = ones(N, J)
z_path_index[Int.(N/2):end, 1] .= 2

for n in 1:N
    for t in 2:J-1
        z_path_index[n,t] = sample(1:nz, Weights(prim.Pr_z[Int.(z_path_index[n, t-1]),:]))
    end
end


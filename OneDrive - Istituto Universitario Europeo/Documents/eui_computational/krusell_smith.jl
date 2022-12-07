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
# using Pckg

using QuantEcon
using BenchmarkTools
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore

Random.seed!(1234)

@with_kw struct Primitives
    
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 1.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    A::Float64 = 1.0

    nz::Int64 = 2
    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Array{Float64, 2} = [0.9 0.1; 0.1 0.9]

    Z_mat::Array{Float64, 2} = [0.5 0.5; 0.1 0.9]
    Z_agg_grid::Vector{Float64} = [0.99, 1.01]

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
function firm_decision(prim::Primitives, Agg_L::Float64)
    
    @unpack α, δ, β, Z_agg_grid, K_agg_grid = prim

    r = α .* Z_agg_grid * K_agg_grid'.^(α-1).* (Agg_L)^(1-α) .- δ
    wage = (1-α).* Z_agg_grid *(K_agg_grid').^(α) .* (Agg_L)^(-α)
    
   return r, wage
end


function util_(prim::Primitives, c)
    @unpack σ = prim
    u = σ == 1 ? x -> log.(x) : x -> (x.^(1 - σ) .- 1) ./ (1 - σ)
    return u(c)
end


function young_2010_continuous(prim::Primitives, kpol)

    @unpack nz, na, trans_mat, a_grid_log, tol, Z_agg_grid, K_agg_grid, n_agg = prim

    a_grid = copy(a_grid_log)
    ind_low = ones(length(Z_agg_grid), n_agg, nz, na)
    for a in 2:na
        for z in 1:nz
            for Z in 1:length(Z_agg_grid)
                for K in 1:n_agg
                    ind_low[Z, K, z, findall(x -> x >= a_grid[a], kpol[Z, K, z,:])] .=  a
                    ind_low[Z, K, z, findall(x -> x >= na, ind_low[Z,K,z,:])] .=  na-1
                end
            end
        end
    end
    
    ind_up = ind_low .+ 1

    wabove = ones(length(Z_agg_grid), n_agg, nz, na)
    wbelow = ones(length(Z_agg_grid), n_agg, nz, na)

    for z in 1:nz
        for i in 1:na
            for Z in 1:length(Z_agg_grid)
                for K in 1:n_agg
                    wabove[Z,K,z,i] =  (kpol[Z,K,z,i] - a_grid[Int(ind_low[Z,K,z,i])]) / (a_grid[Int(ind_low[Z,K,z,i]) + 1] - a_grid[Int(ind_low[Z,K,z,i])])
                    wabove[Z,K,z,i] = min(wabove[Z,K,z,i], 1)
                    wabove[Z,K,z,i] = max(wabove[Z,K,z,i], 0)
                    wbelow[Z,K,z,i] = 1-wabove[Z,K,z,i]
                end
            end
        end
    end 

    Γ = zeros(length(Z_agg_grid), n_agg, nz*na, nz*na)
    for z in 1:nz
        for i in 1:na
            for Z in 1:length(Z_agg_grid)
                for K in 1:n_agg
                    Γ[Z, K, Int.((i-1)*nz+z), Int.((ind_low[Z,K,z,i]-1)*nz+1 : ind_low[Z,K,z,i]*nz)] = wbelow[Z,K,z,i]*trans_mat[z,:]
                    Γ[Z, K, Int.((i-1)*nz+z), Int.((ind_up[Z,K,z,i]-1)*nz+1 : ind_up[Z,K,z,i]*nz)] = wabove[Z,K,z,i]*trans_mat[z,:]
                end
            end
        end
    end

    probst1 = zeros(length(Z_agg_grid), n_agg, nz*na)
    err = 1

    for Z in 1:length(Z_agg_grid)
        for K in 1:n_agg

            probst = (1/(nz*na)).* ones(nz*na)
            err = 1

            while err > 1e-10
                probst1[Z,K,:] = Γ[Z,K,:,:] ⊡ probst
                err = maximum(abs.(probst1[Z,K,:,:] - probst))
                probst = copy(probst1[Z,K,:,:])
            end
        end
    end

    dist_fin = reshape(probst1, (length(Z_agg_grid), n_agg, nz, na))

    return dist_fin
end


function agg_capital_hh(prim::Primitives, dist_fin::Matrix{Float64}, Kg::Matrix{Float64}, Agg_L::Float64, Z_now::Float64)
    
    @unpack A, α = prim

    agg_temp = sum(dist_fin.* Kg)

    Y = Z_now*(sum(agg_temp))^(α) * (Agg_L)^(1 - α)

    return agg_temp, Y
end


function EGM(prim::Primitives, wage::Float64, r::Float64)

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_log, tol, n_agg, Z_agg_grid, Z_mat, K_agg_grid, maxiter = prim

    a_grid = copy(a_grid_log)
    kpol_egm = zeros(nz, na)
    cpol = ones(nz, na)
    err  = 1
    count = 0

    while err > tol && count < maxiter

        count = count + 1

        Ec = Z_mat ⊡ (trans_mat ⊡ cpol.^(-σ))
        
        c_impl = ((1 .+ r).*β.*Ec).^(-1/σ)
        
        k_impl =  (1 + r)^(-1) .* (c_impl .+ ones(nz, 1) ⊡ collect(a_grid') .- (z_grid ⊡ wage .* ones(1, na)))
                    
        # Interpolation: i only understand it with vectors => go to vector level (a_grid)
        for (z,_) in enumerate(z_grid)

            #for (K_idx,_) in enumerate(K_agg_grid)

             #   for (Z_idx,_) in enumerate(Z_agg_grid)
                    
                    nodes = (vec(k_impl[z,:]),) # Define the nodes
                    itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
                    etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
                    kpol_egm[z,:] = etpf(a_grid) # Perform extrapolation

               # end
            #end
        end

        # Make sure boundaries are kept
        kpol_egm[(kpol_egm .< 0)] .= 0
        kpol_egm[(kpol_egm .> a_grid[na])] .= a_grid[na]

        # Back out implied consumption and check for convergence
        cpol1 = (1 .+ r) .* ones(nz, 1) ⊡ collect(a_grid') - kpol_egm + (z_grid ⊡ wage .* ones(1, na))
        err = maximum(abs.(cpol-cpol1))
        cpol = deepcopy(cpol1)


        println("err = ", round.(err; digits=6)," at iteration ", count)
    end

    return kpol_egm, cpol
end

function EGM_transition(prim::Primitives, wage::Float64, r::Float64, r_next::Float64, cpol::Matrix{Float64})

    # Here, we do not find convergence. We just take the previous results and compute the responses.
    # This gives inputs for next period. We simulate the economy sequentionally.

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_log  = prim
    
    a_grid = copy(a_grid_log)

    c_impl = ((1+r) * β * Z_mat ⊡ (trans_mat ⊡ cpol.^(-σ))).^(-1/σ)
    
    k_impl =  (c_impl + ones(nz,1)*a_grid' - z_grid*wage*ones(1,na))./(1+r)
    kpol_egm_trans = zeros(nz, na)

    for (z,_) in enumerate(z_grid)
        nodes = (vec(k_impl[z,:]),) # Define the nodes
        itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
        etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
        kpol_egm_trans[z,:] = etpf(a_grid) # Perform extrapolation
    end

    # Make sure boundaries are kept
    kpol_egm_trans[(kpol_egm_trans .< 0)] .= 0
    kpol_egm_trans[(kpol_egm_trans .> a_grid[na])] .= a_grid[na]


    cpol1_trans = (1+r)*ones(nz,1).*a_grid' - kpol_egm_trans[:,:] + z_grid*wage*ones(1,na) 


    return kpol_egm_trans, cpol1_trans
end


function sim_Z_shocks(prim::Primitives, n_sim::Int64)

    return simulate(MarkovChain(prim.Z_mat), n_sim),  simulate(MarkovChain(prim.trans_mat), n_sim)
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


function time_series_k(prim, kpol, n_sim, Z_index, z_index, K_index)

    @unpack nz, na, Z_agg_grid, n_agg, a_grid_log = prim

    a_grid = copy(a_grid_log)
    # Start with 0 capital and associated index

    a_sim = zeros(n_sim)
    for t = 1:2
        t = 2
        nodes       = (a_grid,)
        itp         = interpolate(nodes, kpol[Int(Z_index[t]), Int(K_index[t]), Int(z_index[t]), :], Gridded(Linear()))
        extrp       = extrapolate(itp,Line())
        a_sim[t+1]  = extrp(a_sim[t])
    end

    return 
end


function solve_general_model(n_sim)

    count = 0
    error = 1

    prim = Primitives()
    agg_k_hh = 0
    Agg_K_hh = zeros(length(prim.Z_agg_grid), n_agg)
    kpol = zeros(length(prim.Z_agg_grid), n_agg, prim.nz, prim.na)
    cpol = zeros(length(prim.Z_agg_grid), n_agg, prim.nz, prim.na)
    wage = 0
    #dist = zeros(length(prim.Z_agg_grid), n_agg, prim.nz, prim.na)
    k_firm = 0
    _, L_Agg = agg_labour(prim)

    # Guess ALM
    β₀ₗ, β₁ₗ, β₀ₕ, β₁ₕ = 0.1, 0.1, 0.9, 0.9 

    while error > prim.tol && count < prim.maxiter

        count = count + 1
        
        # Capital Demand & Wage
        r, wage = firm_decision(prim, L_Agg)

        for (K_idx, _) in enumerate(prim.K_agg_grid)

            for (Z_idx, _) in enumerate(prim.Z_agg_grid)
                # Endogenous Grid Method
                kpol[Z_idx, K_idx,:,:], cpol[Z_idx, K_idx,:,:] = EGM(prim, wage[Z_idx, K_idx], r[Z_idx, K_idx])
            end

        end

        Z_index, z_index = sim_Z_shocks(prim, n_sim)
        
        K_index = K_state(Z_index, β₀ₗ, β₁ₗ, β₀ₕ, β₁ₕ, n_sim)
        #dist[:, :, :, :] = young_2010_continuous(prim, kpol[:, :, :, :])


        println("error = ", round.(error; digits=6)," at iteration ", count," with r_egm_1 = ", round.(r_1; digits=4)," and r = ",round.(r; digits=4))                 
        println("Aggregate Capital Supply = ", round(agg_k_hh; digits=4), " with aggregate Capital Demand = ", round(k_firm; digits=4))          

    end

    return dist, kpol, agg_k_hh, r, wage
end





# We are interested in kpol and agg_k_hh
dist, kpol, agg_k_hh, r, wage = solve_general_model(0.03)


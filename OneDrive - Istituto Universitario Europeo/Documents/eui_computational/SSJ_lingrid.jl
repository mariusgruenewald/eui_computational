# This Script replicates Sequence Space Jacobian Method
# Problem Set 1, Computational Methods
# By Marius Gruenewald

#----------------#
# Importing Pckg #
#----------------#

#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("Interpolations")
#import Pkg; Pkg.add("LaTeXStrings")
#import Pkg; Pkg.add("Parameters")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("QuantEcon")
#import Pkg; Pkg.add("BenchmarkTools")
#import Pkg; Pkg.add("StatsBase")

using LinearAlgebra
using QuantEcon
using BenchmarkTools
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using Optim

@with_kw struct Parameter

    minval::Float64 = 0.0
    maxval::Float64 = 30.0
    na::Int64 = 200
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 2.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    A::Float64 = 1.0
    nz::Int64 = 2
    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Matrix{Float64} = [0.9 0.1; 0.1 0.9]
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1
    sim_periods::Int64 = 100
    ϵ::Float64 = 0.0001
    tol::Float64 = 1e-5
    maxiter::Int64 = 250
    
end

function agg_labour(prim::Parameter)
    
    @unpack nz, trans_mat, z_grid, tol = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    return Phi_sd, L_Agg[1]
end

function firm_decision(prim::Parameter, Agg_L::Float64, r) # don't pre specify r so that it can handle scalar and vector, what hierarchy to use?
    
    @unpack α, δ, β, A = prim

    k_firm =  ( (r .+ δ)./((α.*A).*(Agg_L)^(1-α)) ).^(1/(α-1))
    wage = (1-α).*(A.*(α./(r .+ δ)).^ α).^(1/(1-α))
    
   return k_firm, wage
end

# Call this function to compute steady state
function EGM(prim::Parameter, wage::Float64, r::Float64)

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_lin, tol  = prim
    
    a_grid = copy(a_grid_lin)
    kpol_egm = zeros(nz,na)
    cpol = ones(nz,na)
    err  = 1
    #cpol = (z_grid'*wage .+ r*a_grid)
    while err > tol

        Ec = trans_mat * cpol.^(-σ)
        
        c_impl = ((1+r)*β*Ec).^(-1/σ)
        
        k_impl =  (c_impl + ones(nz,1)*a_grid' - z_grid*wage*ones(1,na))./(1+r)
                    
        for (z,_) in enumerate(z_grid)
            nodes = (vec(k_impl[z,:]),) # Define the nodes
            itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
            etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
            kpol_egm[z,:] = etpf(a_grid) # Perform extrapolation
        end

        # Make sure boundaries are kept
        kpol_egm[(kpol_egm .< 0)] .= 0
        kpol_egm[(kpol_egm .> a_grid[na])] .= a_grid[na]


        cpol1 = (1+r)*ones(nz,1).*a_grid' - kpol_egm[:,:] + z_grid*wage*ones(1,na) 
        err = maximum(abs.(cpol-cpol1))
        cpol = copy(cpol1)

    end

    return kpol_egm, cpol
end

# Call this function to compute steady state
function young_2010_continuous(prim::Parameter, kpol::Matrix{Float64})
    @unpack nz, na, trans_mat, a_grid_lin, tol = prim
    a_grid = copy(a_grid_lin)
    ind_low = ones(nz,na)
    for a in 2:na
        for z in 1:nz
            ind_low[z,findall(x -> x >= a_grid[a], kpol[z,:])] .=  a
            ind_low[z,findall(x -> x >= na, ind_low[z,:])] .=  na-1
        end
    end
    
    ind_up = ind_low .+ 1

    wabove = ones(nz, na)
    wbelow = ones(nz, na)

    for z in 1:nz
        for i in 1:na
            wabove[z,i] =  (kpol[z,i] - a_grid[Int(ind_low[z,i])]) / (a_grid[Int(ind_low[z,i]) + 1] - a_grid[Int(ind_low[z,i])])
            wabove[z,i] = min(wabove[z,i],1)
            wabove[z,i] = max(wabove[z,i],0)
            wbelow[z,i] = 1-wabove[z,i]
        end
    end 

    Γ = zeros(nz*na, nz*na)
    for z in 1:nz
        for i in 1:na
            Γ[Int.((i-1)*nz+z), Int.((ind_low[z,i]-1)*nz+1:ind_low[z,i]*nz)] = wbelow[z,i]*trans_mat[z,:]
            Γ[Int.((i-1)*nz+z), Int.((ind_up[z,i]-1)*nz+1:ind_up[z,i]*nz)] = wabove[z,i]*trans_mat[z,:]
        end
    end

    probst = (1/(nz*na))*ones(nz*na)'
    err = 1 
    while err > 1e-10
       probst1 = probst*Γ          
       err = maximum(abs.(probst1-probst))
       probst = copy(probst1)
    end
    dist_fin = reshape(probst, (nz, na))

    return dist_fin
end


# Call this function for transition
function young_2010_transition_SSJ(prim::Parameter, kpol::Matrix{Float64}, dist::Matrix{Float64})
 
    # Needs to be run for every period
    @unpack nz, na, trans_mat, a_grid_lin = prim
    a_grid = copy(a_grid_lin)
    # Locate policy function in the grid, i.e. find indeces above and below
    ind_low = ones(nz,na)
    for a in 2:na
        for z in 1:nz
            ind_low[z,findall(x -> x >= a_grid[a], kpol[z,:])] .=  a
            ind_low[z,findall(x -> x >= na, ind_low[z,:])] .=  na-1
        end
    end
    
    ind_up = ind_low .+ 1

    wabove = ones(nz, na)
    wbelow = ones(nz, na)
    
    # However, policy functions may not be on the grid. Assign share of distribution to gridpoints, what percent of population 
    # will end up above or below policy function. Shouldn't change much => if it does, make grid finer.
    for z in 1:nz
        for i in 1:na
            wabove[z,i] =  (kpol[z,i] - a_grid[Int(ind_low[z,i])]) / (a_grid[Int(ind_low[z,i]) + 1] - a_grid[Int(ind_low[z,i])])
            wabove[z,i] = min(wabove[z,i],1)
            wabove[z,i] = max(wabove[z,i],0)
            wbelow[z,i] = 1-wabove[z,i]
        end
    end 

    # Compute distributional matrix. Here, we do not find invariant distribution but just iteration one period further
    Γ = zeros(nz, na)
    for z in 1:nz
        for i in 1:na
            # Based on matrix today, where will the lower index be tomorrow
            Γ[:, Int.(ind_low[z,i])] = Γ[:, Int.(ind_low[z,i])] .+ dist[z,i]*wbelow[z,i]*trans_mat[z,:]
            # Based on matrix today, where will the upper index be tomorrow
            Γ[:, Int.(ind_up[z,i])] = Γ[:, Int.(ind_up[z,i])] .+ dist[z,i]*wabove[z,i]*trans_mat[z,:]
        end
    end

    return Γ
end

# Call this function for transition
function EGM_transition_SSJ(prim::Parameter, wage::Float64, r::Float64, r_next::Float64, cpol::Matrix{Float64})

    # Here, we do not find convergence. We just take the previous results and compute the responses.
    # This gives inputs for next period. We simulate the economy sequentionally.

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_lin  = prim
    
    a_grid = copy(a_grid_lin)

    c_impl = ((1+r_next) * β * trans_mat * cpol.^(-σ)).^(-1/σ)
    
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

function solve_general_model_SSJ(prim::Parameter, r::Float64)

    @unpack a_grid_lin, tol, maxiter, α, A, δ = prim

    count = 0
    error = 1
    
    agg_k_hh = 0
    Agg_K_hh = 0
    wage = 0
    dist_egm = 0
    k_firm = 0
    cons_levels = 0
    kpol = 0
    a_grid = copy(a_grid_lin)

    _, L_Agg = agg_labour(prim)

    while error > tol && count < maxiter

        count = count + 1
        
        # Capital Demand & Wage
        k_firm, wage = firm_decision(prim, L_Agg, r)
        # Endogenous Grid Method
        kpol, _ = EGM(prim, wage, r)
        cons_levels = prim.z_grid*wage .+ (1+r)*kpol .- a_grid'
        ## Euler Equation Error
        dist_egm = young_2010_continuous(prim, kpol)

        # Capital Supply and Output
        Agg_K_hh = sum(dist_egm.*kpol)
        agg_k_hh = copy(Agg_K_hh)

        # Back out interest rate 
        r_1 = α * A*(agg_k_hh^(α-1))*(L_Agg^(1-α)) - δ

        error = abs(r_1-r)
        println("error = ", round.(error; digits=6)," at iteration ", count," with r_egm_1 = ", round.(r_1; digits=4)," and r = ",round.(r; digits=4))                 
        println("Aggregate Capital Supply = ", round(agg_k_hh; digits=4), " with aggregate Capital Demand = ", round(k_firm; digits=4))          

        r = 0.95*r + 0.05*r_1
    end

    return dist_egm, kpol, cons_levels, agg_k_hh, r, wage, L_Agg
end

function partial_jacobian_r(prim::Parameter, r_init::Float64, dist_init::Matrix{Float64}, cons_init::Matrix{Float64}, 
    kpol_init::Matrix{Float64}, agg_k_init::Float64, shock::Int64)

    @unpack nz, na, a_grid_lin, A, sim_periods, ϵ = prim
    
    r_path = r_init.*ones(sim_periods)
    r_path[shock] = r_path[shock] + ϵ
    
    # Set up container
    cpol_trans = zeros(nz, na, sim_periods)
    kpol_trans = zeros(nz, na, sim_periods)
    dist_trans = zeros(nz, na, sim_periods)
    agg_k_hh_trans = zeros(sim_periods)
    wage = zeros(sim_periods)

    dist_trans[:,:,1] = dist_init
    # Initialize results from the end (backward shooting)
    cpol_trans[:,:,sim_periods] = cons_init
    kpol_trans[:,:,sim_periods] = kpol_init

    _, Agg_L = agg_labour(prim)
    _, wage = firm_decision(prim, Agg_L, r_path)

    # backwards iteration with cpol and kpol as initiated from the end
    for t in reverse(1:sim_periods-1)
        kpol_trans[:,:,t], cpol_trans[:,:,t] = EGM_transition_SSJ(prim, wage[t], r_path[t], r_path[t+1], cpol_trans[:,:,t])
    end

    # shoot forward
    for t in 1:sim_periods-1
        dist_trans[:,:,t+1] = young_2010_transition_SSJ(prim, kpol_trans[:,:,t], dist_trans[:,:,t])
        agg_k_hh_trans[t+1] = sum(dist_trans[:,:,t] .* kpol_trans[:,:,t])
    end

    partial_k = agg_k_hh_trans .- agg_k_init

    return kpol_trans, cpol_trans, partial_k
end

function partial_jacobian_w(prim::Parameter, wage_init::Float64, dist_init::Matrix{Float64}, cons_init::Matrix{Float64}, 
    kpol_init::Matrix{Float64}, agg_k_init::Float64, shock::Int64)

    @unpack nz, na, a_grid_lin, A, α, δ, sim_periods, ϵ = prim
    
    # Set up path 
    w_path = wage_init.*ones(sim_periods)
    w_path[shock] = w_path[shock] + ϵ
    
    # Set up container
    cpol_trans = zeros(nz, na, sim_periods)
    kpol_trans = zeros(nz, na, sim_periods)
    dist_trans = zeros(nz, na, sim_periods)
    agg_k_hh_trans = zeros(sim_periods)

    dist_trans[:,:,1] = dist_init
    # Initialize results from the end (backward shooting)
    cpol_trans[:,:,sim_periods] = cons_init
    kpol_trans[:,:,sim_periods] = kpol_init

    # Back out interest rate from Firm optimization
    r_path = (1-α)^((1-α)/α) .* (A ./ (w_path).^((1-α)/α) ) .- δ

    # backwards iteration with cpol and kpol as initiated from the end
    for t in reverse(1:sim_periods-1)
        kpol_trans[:,:,t], cpol_trans[:,:,t] = EGM_transition_SSJ(prim, w_path[t], r_path[t], r_path[t+1], cpol_trans[:,:,t])
    end

    # shoot forward
    for t in 1:sim_periods-1
        dist_trans[:,:,t+1] = young_2010_transition_SSJ(prim, kpol_trans[:,:,t], dist_trans[:,:,t])
        agg_k_hh_trans[t+1] = sum(dist_trans[:,:,t] .* kpol_trans[:,:,t])
    end

    partial_k = agg_k_hh_trans .- agg_k_init

    return kpol_trans, cpol_trans, partial_k
end

function closed_form_pj(prim::Parameter, agg_k_init::Float64, L_Agg::Float64)

    @unpack α, A, sim_periods = prim

    # Partial Jacobian of interest rate with respect to capital
    store_pj_rk = zeros(sim_periods, sim_periods)
    store_pj_rk[diagind(store_pj_rk)] .= α * (α-1) * A * agg_k_init^(α-2) * L_Agg^(1-α)

    # Partial Jacobian of interest rate with respect to productivity
    store_pj_rz = zeros(sim_periods, sim_periods)
    store_pj_rz[diagind(store_pj_rz)] .= α * agg_k_init^(α-1) * L_Agg^(1-α)

    # Partial Jacobian of wages with respect to capital
    store_pj_wk = zeros(sim_periods, sim_periods)
    store_pj_wk[diagind(store_pj_wk)] .= α * (1-α) * A * agg_k_init^(α-1) * L_Agg^(-α)

    # Partial Jacobian of wages with respect to productivity
    store_pj_wz = zeros(sim_periods, sim_periods)
    store_pj_wz[diagind(store_pj_wz)] .= (1-α) * agg_k_init^(α) * L_Agg^(-α)

    return store_pj_rk, store_pj_rz, store_pj_wk, store_pj_wz
end

function get_jacobians()

    prim = Parameter()
    # Run again to have stuff inside loop
    @time dist_init, kpol_init, cons_init, agg_k_init, r_init, wage_init, L_Agg = solve_general_model_SSJ(prim, 0.03)

    store_pj_kr = zeros(prim.sim_periods, prim.sim_periods)
    store_pj_kw = zeros(prim.sim_periods, prim.sim_periods)
    for t in 1:prim.sim_periods 
        _, _, store_pj_kr[:,t] = partial_jacobian_r(prim, r_init, collect(dist_init), cons_init, kpol_init, agg_k_init, t)
        _, _, store_pj_kw[:,t] = partial_jacobian_w(prim, wage_init, collect(dist_init), cons_init, kpol_init, agg_k_init, t)
        println("Computing Jacobians at period: ", t)                 
    end

    store_pj_rk, store_pj_rz, store_pj_wk, store_pj_wz = closed_form_pj(prim, agg_k_init, L_Agg)

    return store_pj_kr, store_pj_kw, store_pj_rk, store_pj_rz, store_pj_wk, store_pj_wz 
end

function compute_h(store_pj_kr::Matrix{Float64}, store_pj_kw::Matrix{Float64}, store_pj_rk::Matrix{Float64}, 
    store_pj_rz::Matrix{Float64}, store_pj_wk::Matrix{Float64}, store_pj_wz::Matrix{Float64})

    # Compute total jacobians
    H_k = store_pj_kr*store_pj_rk + store_pj_kw*store_pj_wk - I
    H_z = store_pj_kr*store_pj_rz + store_pj_kw*store_pj_wz

    return H_k, H_z
end

function solve_SSJ(shock::Float64, period::Int64)

    prim = Parameter()
    # Compute Partial Jacobians
    store_pj_kr, store_pj_kw, store_pj_rk, store_pj_rz, store_pj_wk, store_pj_wz = get_jacobians()

    H_k, H_z = compute_h(store_pj_kr, store_pj_kw, store_pj_rk, store_pj_rz, store_pj_wk, store_pj_wz)

    G_kz = -inv(H_k)*H_z
    dZ = zeros(prim.sim_periods)
    dZ[period] = shock
    dK = G_kz*dZ

    return dK
end

dK = solve_SSJ(-0.1, 10)

# Visualize IRF
IRF_capital = plot(1:100, dK, label = "IRF of Capital", dpi=300, title = "Temporary shock to productivity")
xaxis!("Periods")
yaxis!("Change")
savefig(IRF_capital,"IRF_capital.png")
# This looks super weird and different from the extended path method. I don't know why. :(
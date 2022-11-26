# This Script denotes transition dynamics in an Aiyagari Economy
# Problem Set 1, Computational Methods
# By Marius Gruenewald


using QuantEcon
using BenchmarkTools
using CSV
using DataFrames
using DelimitedFiles
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase

Random.seed!(1234)

@with_kw struct Primitives
    
    minval::Float64 = 0.0
    maxval::Float64 = 50.0
    na::Int64 = 200
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 2.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    nz::Int64 = 2
    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Matrix{Float64} = [0.1 0.9; 0.9 0.1]
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

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

function firm_decision(prim::Primitives, Agg_L::Float64, r, A_path)
    
    @unpack α, δ, β = prim

    k_firm =  ( (r .+ δ)./((α.*A_path).*(Agg_L)^(1-α)) ).^(1/(α-1))
    wage = (1-α).*(A_path.*(α./(r .+ δ)).^ α).^(1/(1-α))
    
   return k_firm, wage
end

# Call this function to compute steady state
function EGM(prim::Primitives, wage::Float64, r::Float64)

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_lin  = prim
    
    a_grid = copy(a_grid_lin)
    kpol_egm = zeros(nz,na)
    cpol = ones(nz,na)
    tol_pol = 0.00001
    err  = 1
    #cpol = (z_grid'*wage .+ r*a_grid)
    while err > tol_pol

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
function young_2010_continuous(prim::Primitives, kpol::Matrix{Float64})
    @unpack nz, na, trans_mat, a_grid_lin = prim
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
    while err > 1e-6  
       probst1 = probst*Γ          
       err = maximum(abs.(probst1-probst))
       probst = copy(probst1)
    end
    dist_fin = reshape(probst, (nz, na))

    return dist_fin
end

# Call this function for transition
function young_2010_transition(prim::Primitives, kpol::Matrix{Float64}, dist::Matrix{Float64})
 
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
function EGM_transition(prim::Primitives, wage::Float64, r::Float64, r_next::Float64, cpol::Matrix{Float64})

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

# Run twice for permanent shocks => old and new steady state
# Idea for transitory shock: two old and new steady state. Calculate two transition path, one for period with trans. shock 
# the other transition from shock back to equilibrium.
function solve_general_model(r::Float64, A::Float64)

    count = 0
    error = 1
    tol_egm = 1e-5
    maxiter = 100
    prim = Primitives()
    agg_k_hh = 0
    Agg_K_hh = 0
    wage = 0
    dist_egm = 0
    k_firm = 0
    cons_levels = 0
    kpol = 0
    a_grid = copy(prim.a_grid_lin)
    #minrate     = prim.δ
    #maxrate     = (1-prim.β)/(prim.β)

    _, L_Agg = agg_labour(prim)

    while error > tol_egm && count < maxiter

        count = count + 1
        
        # Capital Demand & Wage
        k_firm, wage = firm_decision(prim, L_Agg, r, A)
        # Endogenous Grid Method
        kpol, _ = EGM(prim, wage, r)
        cons_levels = prim.z_grid*wage .+ (1+r)*kpol .- a_grid'
        ## Euler Equation Error
        dist_egm = young_2010_continuous(prim, kpol)

        # Capital Supply and Output
        Agg_K_hh = sum(dist_egm.*kpol)
        agg_k_hh = copy(Agg_K_hh)

        # Back out interest rate 
        r_1 = prim.α * A*(agg_k_hh^(prim.α-1))*(L_Agg^(1-prim.α)) - prim.δ

        error = abs(r_1-r)
        println("error = ", round.(error; digits=6)," at iteration ", count," with r_egm_1 = ", round.(r_1; digits=4)," and r = ",round.(r; digits=4))                 
        println("Aggregate Capital Supply = ", round(agg_k_hh; digits=4), " with aggregate Capital Demand = ", round(k_firm; digits=4))          

        r = 0.95*r + 0.05*r_1
    end

    return dist_egm, kpol, cons_levels, agg_k_hh, r, wage
end

dist_init, kpol_init, cons_init, agg_k_init, r_init, wage_init = solve_general_model(0.03, 1.0)
dist_final, kpol_final, cons_final, agg_k_final, r_final, wage_final = solve_general_model(0.03, 0.9)

function transition(prim::Primitives, r_init::Float64, r_final::Float64, transition_periods::Int64, dist_init::Matrix{Float64}, 
    cons_final::Matrix{Float64}, kpol_final::Matrix{Float64}, A_init::Float64, A_final::Float64, shock_perm::Bool)

    @unpack nz, na, a_grid_lin, α, δ = prim

    if shock_perm == true
        A_path = A_init*ones(transition_periods)
        A_path[2:end] = A_final.*A_path[2:end]
    else
        A_path = A_init*ones(transition_periods)
        A_path[2] = A_final*A_path[2]
    end

    maxiter = 500
    a_grid = copy(a_grid_lin)
    # Set up container
    r = collect(LinRange(r_init, r_final, transition_periods)) # intial guess of path 
    dist_trans = zeros(nz, na, transition_periods)
    cpol_trans = zeros(nz, na, transition_periods)
    kpol_trans = zeros(nz, na, transition_periods)
    agg_k_hh_trans = zeros(transition_periods)
    agg_c_trans = zeros(transition_periods)
    wage = zeros(transition_periods)

    # Initialize results from the beginning (forward shooting)
    dist_trans[:,:,1] = dist_init
    agg_k_hh_trans[1] = sum(dist_init .* (ones(nz)*a_grid'))

    # Initialize results from the end (backward shooting)
    cpol_trans[:,:,transition_periods] = cons_final
    kpol_trans[:,:,transition_periods] = kpol_final

    _, Agg_L = agg_labour(prim)

    iter = 0
    error = 1
    tol = 1e-4
    while error > tol && iter < maxiter

        # intiate wage for EGM
        _, wage = firm_decision(prim, Agg_L, r, A_path)

        # backwards iteration with cpol and kpol as initiated from the end
        for t in reverse(1:transition_periods-1)
            kpol_trans[:,:,t], cpol_trans[:,:,t] = EGM_transition(prim, wage[t], r[t], r[t+1], cpol_trans[:,:,t])
        end
        
        # forward shooting for distribution and consumption as initiated from the beginning

        dist_trans[:,:,2:transition_periods] .= 0  # reset to zeros (important as cumulative method!)
        # Initiate consumption
        agg_c_trans[1] = sum(dist_trans[:,:,1].*cpol_trans[:,:,1])
        # shoot forward
        for t in 1:transition_periods-1
            dist_trans[:,:,t+1] = young_2010_transition(prim, kpol_trans[:,:,t], dist_trans[:,:,t])
            agg_k_hh_trans[t+1] = sum(dist_trans[:,:,t] .* kpol_trans[:,:,t])
            agg_c_trans[t+1] = sum(dist_trans[:,:,t+1] .* cpol_trans[:,:,t+1])
        end

        r_new = A_path .* α .* agg_k_hh_trans.^(α-1) * Agg_L^(1-α) .-δ
        error = maximum(abs.(r - r_new))
        println("error = ", round.(error; digits=6)," at iteration ", iter)
        r = 0.95 .*r .+ 0.05.*r_new
        iter = iter + 1
    end

    return r, wage, dist_trans, agg_k_hh_trans, agg_c_trans, kpol_trans, cpol_trans
end

r_perm, wage_perm, dist_trans_perm, agg_k_trans_perm, agg_c_trans_perm, kpol_perm, cpol_perm = transition(prim, r_init, r_final, 100, collect(dist_init), cons_final, kpol_final, 1.0, 0.9, true)

perm_trans_r = plot(1:100, r_perm, legend=false, dpi=300, title = "Transition Path to Permanent Shock - Interest Rate")
xaxis!("Time")
yaxis!("Interest Rate")
savefig(perm_trans_r,"perm_trans_r.png")

perm_trans_w = plot(1:100, wage_perm, legend=false, dpi=300, title = "Transition Path to Permanent Shock - Wage")
xaxis!("Time")
yaxis!("Wage")
savefig(perm_trans_w,"perm_trans_w.png")

perm_trans_cap = plot(1:100, agg_k_trans_perm, legend=false, dpi=300, title = "Transition Path to Permanent Shock - Agg. Capital")
xaxis!("Time")
yaxis!("Aggregate Capital Supply")
savefig(perm_trans_cap,"perm_trans_cap.png")

perm_trans_cons = plot(1:100, agg_c_trans_perm, legend=false,dpi=300, title = "Transition Path to Permanent Shock - Consumption")
xaxis!("Time")
yaxis!("Aggregate Consumption")
savefig(perm_trans_cons,"perm_trans_cons.png")


##### Temporary Shock

r_temp, wage_temp, dist_trans_temp, agg_k_trans_temp, agg_c_trans_temp, kpol_temp, cpol_temp = transition(prim, r_init, r_final, 100, collect(dist_init), cons_final, kpol_final, 1.0, 0.9, false)

temp_trans_r = plot(1:100, r_temp, legend=false, dpi=300, title = "Transition Path to Temporary Shock - Interest Rate")
xaxis!("Time")
yaxis!("Interest Rate")
savefig(temp_trans_r,"temp_trans_r.png")

temp_trans_w = plot(1:100, wage_temp, legend=false, dpi=300, title = "Transition Path to Temporary Shock - Wage")
xaxis!("Time")
yaxis!("Wage")
savefig(temp_trans_w,"temp_trans_w.png")

temp_trans_cap = plot(1:100, agg_k_trans_temp, legend=false, dpi=300, title = "Transition Path to Temporary Shock - Agg. Capital")
xaxis!("Time")
yaxis!("Aggregate Capital Supply")
savefig(temp_trans_cap,"temp_trans_cap.png")

temp_trans_cons = plot(1:100, agg_c_trans_temp, legend=false,dpi=300, title = "Transition Path to Temporary Shock - Consumption")
xaxis!("Time")
yaxis!("Aggregate Consumption")
savefig(temp_trans_cons,"temp_trans_cons.png")

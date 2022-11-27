# This is a replication attempt at the Aiyagari Model with VFI and EGM
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

Random.seed!(1234)

@with_kw struct Primitive
    
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
    A::Float64 = 1.0
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1
    tol::Float64 = 1e-5
    maxiter::Int64 = 500
end

function agg_labour(prim::Primitive)
    
    @unpack nz, trans_mat, z_grid, tol = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    # multiply productivity with labour share 

    return Phi_sd, L_Agg[1]
end

function firm_decision(prim::Primitive, Agg_L::Float64, r::Float64)
    
    @unpack α, A, δ, β = prim

    k_firm =  ( (r + δ)/((α*A)*(Agg_L)^(1-α)) )^(1/(α-1))
    wage = (1-α)*A*(k_firm)^(α)* (Agg_L)^(-α)
    
   return k_firm, wage
end

function fcons(prim::Primitive, wage::Float64, a::Float64, r::Float64)

    a_grid = copy(prim.a_grid_lin)
    a_grid_dim = repeat(a_grid, 1, prim.nz)
    return wage*prim.z_grid .+ (1+r)*a .- a_grid_dim'
end


function util_(prim::Primitive, c::Matrix{Float64})
    return (c.^(1-prim.σ))./(1-prim.σ)
end


function VFI(prim::Primitive, wage::Float64, r::Float64)
    
    @unpack trans_mat, β, σ, na, nz, z_grid, a_grid_lin, tol = prim

    a_grid = copy(a_grid_lin)
    v_now = zeros(nz, na)
    v_next = zeros(nz, na)
    policy = zeros(nz, na)
    error = 1
    util = 0
    c = 0
    
    while error > tol
        
        for a in 1:na
            
            c = fcons(prim, wage, a_grid[a], r) # calculate consumption
            util = util_(prim, c) # calculate util
            util[findall(x-> x<=0, c)] .= -1e8 # replace unreasonable combinations
            
            v_temp = util + β.* trans_mat * v_next # value function
            v_now[:,a] = maximum(v_temp, dims=2) # find max
            
            for z in 1:nz
                policy[z, a] = findmax(v_temp, dims=2)[2][z][2] # policy decision
            end
        
        end
        error = maximum(abs.(v_now - v_next))
        v_next = copy(v_now)
    end
    # set up capital-state x capita-state matrix that incorporates the prob of being in a certain capital-state point
    # tomorrow based on today's decision
    return policy, v_now
end


function EGM(prim::Primitive, wage::Float64, r::Float64)

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

# not working as of now
function calc_EEE(prim::Primitive, na_precise::Int64, kpol::Matrix{Float64}, r::Float64, wage::Float64)
    
    @unpack minval, maxval, nz, na, β, σ, trans_mat, z_grid, a_grid_lin = prim
    # make fine grid
    a_grid = copy(a_grid_lin)
    f_grid = zeros(na_precise)
    f_grid = collect(range(minval, maxval, length=na_precise))

    # interpolate g_pol at nkk_precise points 
    EE_error  = zeros(nz, na_precise)
    g_prec = zeros(nz, na_precise)
    c = zeros(nz, na_precise)
    Ec_p = zeros(nz, na_precise)

    for z = 1:nz

        nodes   = (a_grid,)
        itp     = interpolate(nodes, kpol[z,:], Gridded(Linear()))
        extrp  = extrapolate(itp,Line())
        g_prec[z,:] = extrp(f_grid)

        g_p_prec    = zeros(nz,na_precise)
        for z_p = 1:nz
            # interpolate the future decision
            nodes_g  = (a_grid,)
            itp_g    = interpolate(nodes_g, kpol[z_p,:], Gridded(Linear()))
            extrp_g  = extrapolate(itp_g,Line())
            g_p_prec[z_p,:]=extrp_g(g_prec[z,:])
        end

        for a = 1:na_precise
            # consumption today
            c[z,a]  = f_grid[a]*(1+r) + z*wage - g_prec[z,a]
            #consumption tomorrow
            Ec_p[z,a] = β*(1+r)*trans_mat[z,:]'*(g_prec[z,a]*(1+r)*ones(nz) + z_grid*wage - g_p_prec[:,a])
            # use formula to calc realtive EEE in terms of consumption 
            EE_error[z,a] = log(abs((c[z,a]/Ec_p[z,a]).^(-σ) - 1))
        end

    end
    
    return EE_error, f_grid
end

function young_2010_continuous(prim::Primitive, kpol::Matrix{Float64})

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

function young_2010_discrete(prim::Primitive, policy_a::Matrix{Float64})

    @unpack nz, na, trans_mat, tol = prim

    Γ = zeros(nz*na, nz*na)

    for z in 1:nz
        for i in 1:na
            Γ[((i-1)*nz+z), Int.((policy_a[z,i]-1)*nz + 1:policy_a[z,i]*nz)] = trans_mat[z,:]
        end
    end

    probst = (1/(nz*na))*ones(1,nz*na)
    err = 1                      
    while err > 1e-10
       probst1 = probst*Γ          
       err = maximum(abs.(probst1-probst))
       probst = probst1
    end
    dist_fin = reshape(probst, (nz, na))

    return dist_fin
end

function agg_capital_hh(prim::Primitive, dist_fin::Matrix{Float64}, Kg::Matrix{Float64}, Agg_L::Float64)
    
    @unpack A, α = prim

    agg_temp = sum(dist_fin.*Kg)

    Y = A*(sum(agg_temp))^(α) * (Agg_L)^(1 - α)

    return agg_temp
end

function solve_partial_model(VFI_bool::Bool)

    prim = Primitive()
    #Phi_sd, L_Agg = agg_labour(prim)

    r = 0.03
    wage = 1.3

    if r <= 0
        blim = 0
    else
        blim = maximum([0, -(wage*prim.z_grid[1])/r])
        if blim == 0
            println("No natural borrowing constraint binds, only ad hoc of 0")
        else
            println("Natural borrowing constraint binds => Check bounds of capital grid")
        end
    end

    #a_grid = asset_grid(prim, wage, r)
    if VFI_bool == true
        # Value Function Limbo
        policy_a, v_now = VFI(prim, wage, r)
        # Consumption Levels
        cons_levels = prim.z_grid*wage .+ (1+r)*policy_a .- prim.a_grid_lin'
        ## Euler Equation Error
        EE_error_vfi, f_grid_vfi = calc_EEE(prim, 10000, policy_a, r, wage)
        return policy_a, v_now, cons_levels, EE_error_vfi, f_grid_vfi

    else
        # Endogenous Grid Method
        kpol, cpol = EGM(prim, wage, r)
        ## Euler Equation Error
        EE_error_egm, f_grid_egm = calc_EEE(prim, 10000, kpol, r, wage)
        return kpol, cpol, EE_error_egm, f_grid_egm 

    end
end

#### Value Function Iteration - Something is off here, but what?
@time policy_a, v_now, cons_levels, EE_error_vfi, f_grid_vfi = solve_partial_model(true)


gr()
prim = Primitive()
value_vfi = plot(prim.a_grid_lin, [v_now[1,:], v_now[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Value Functions")
xaxis!(L"a")
yaxis!("Value Function")
savefig(value_vfi,"value_vfi.png")

cons_vfi = plot(prim.a_grid_lin, [cons_levels[1,:], cons_levels[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Consumption Functions")
xaxis!(L"a")
yaxis!("Consumption Function")
savefig(cons_vfi,"cons_vfi.png")

pol_vfi = plot(prim.a_grid_lin, [policy_a[1,:], policy_a[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Policy Functions")
xaxis!(L"a")
yaxis!("Policy Function")
savefig(pol_vfi,"pol_vfi.png")

eee_vfi = plot(f_grid_vfi, [EE_error_vfi[1,:], EE_error_vfi[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Euler Equation Error")
xaxis!("Fine assets grid")
yaxis!("Euler Equation error")
savefig(eee_vfi,"eee_vfi.png")

#### Endogenous Grid Method
@time kpol, cpol, EE_error_egm, f_grid_egm  = solve_partial_model(false)

cons_egm = plot(prim.a_grid_lin, [cpol[1,:], cpol[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Consumption Functions")
xaxis!(L"a")
yaxis!("Consumption Function")
savefig(cons_egm,"cons_egm.png")

pol_egm = plot(prim.a_grid_lin, [kpol[1,:], kpol[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Policy Functions")
xaxis!(L"a")
yaxis!("Policy Function")
savefig(pol_egm,"pol_egm.png")

eee_egm = plot(f_grid_egm, [EE_error_egm[1,:], EE_error_egm[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Euler Equation Error")
xaxis!("Fine assets grid")
yaxis!("Euler Equation error")
savefig(eee_egm,"eee_egm.png")

function solve_general_model(VFI_bool::Bool, r::Float64)

    count = 0
    error = 1

    prim = Primitive()
    agg_k_hh = 0
    Agg_K_hh = 0
    wage_egm = 0
    wage = 0
    agg_k_hh_egm = 0
    dist_egm = 0
    dist_vfi = 0
    k_firm_egm = 0

    #minrate     = prim.δ
    #maxrate     = (1-prim.β)/(prim.β)
    
    _, L_Agg = agg_labour(prim)

    if VFI_bool == true

        while error > prim.tol && count < prim.maxiter

            count = count + 1
            # Capital Demand & Wage
            k_firm, wage = firm_decision(prim, L_Agg, r)
            # Value Function Limbo
            policy_a, v_now = VFI(prim, wage, r)
            # Consumption Levels
            cons_levels = prim.z_grid*wage .+ (1+r)*policy_a .- prim.a_grid_lin'
            ## Euler Equation Error
            dist_vfi = young_2010_discrete(prim, policy_a)
            # Capital Supply and Output
            Agg_K_hh = sum(dist_vfi.*policy_a)
            agg_k_hh = copy(Agg_K_hh)

            # Back out interest rate 
            r_1 = prim.α*prim.A*(agg_k_hh^(prim.α-1))*(L_Agg^(1-prim.α)) - prim.δ
            error = abs(r_1-r)
            println("error = ", round.(error; digits=6)," at iteration ", count," with r_1 = ", round.(r_1; digits=4)," and r = ",round.(r; digits=4))
            println("Aggregate Capital Supply = ", round(agg_k_hh; digits=4), " with aggregate Capital Demand = ", round(k_firm; digits=4))          
            
            r = 0.9*r + 0.1*r_1

        end

        return dist_vfi, policy_a, agg_k_hh, r, wage

    else

        while error > prim.tol && count < prim.maxiter

            count = count + 1
            
            # Capital Demand & Wage
            k_firm_egm, wage_egm = firm_decision(prim, L_Agg, r)
            # Endogenous Grid Method
            kpol, cpol = EGM(prim, wage_egm, r)
            ## Euler Equation Error
            dist_egm = young_2010_continuous(prim, kpol)

            # Capital Supply and Output
            Agg_K_hh_egm = sum(dist_egm.*kpol)
            agg_k_hh_egm = copy(Agg_K_hh_egm)

            # Back out interest rate 
            r_egm_1 = prim.α * prim.A*(agg_k_hh_egm^(prim.α-1))*(L_Agg^(1-prim.α)) - prim.δ

            error = abs(r_egm_1-r)
            println("error = ", round.(error; digits=6)," at iteration ", count," with r_egm_1 = ", round.(r_egm_1; digits=4)," and r = ",round.(r; digits=4))                 
            println("Aggregate Capital Supply = ", round(agg_k_hh_egm; digits=4), " with aggregate Capital Demand = ", round(k_firm_egm; digits=4))          

            r = 0.95*r + 0.05*r_egm_1
        end

        return dist_egm, kpol, agg_k_hh_egm, r, wage_egm
        
    end
end
#### Value Function Iteration, does not quite work
@time dist_vfi, policy_a, agg_k_hh_vfi, r_vfi, wage_vfi  = solve_general_model(true, 0.3)
dist_vfi_plot = plot(prim.a_grid_lin, dist_vfi', label = [L"z_1" L"z_2"], dpi=300, title = "Probability of Assets by Productivity VFI")
xaxis!("Assets")
yaxis!("Probability")
savefig(dist_vfi_plot,"dist_vfi.png")

ergodic_dist_vfi = plot(prim.a_grid_lin, dist_vfi'.*policy_a', label = [L"z_1" L"z_2"], dpi=300, title = "Distribution of Assets by Productivity VFI")
xaxis!("Assets")
yaxis!("Distribution")
savefig(ergodic_dist_vfi,"ergodic_dist_vfi.png")


# Seems to work for very cautious updates
@time dist_egm, kpol, agg_k_hh_egm, r_egm, wage_egm  = solve_general_model(false, 0.3)
dist_egm_plot = plot(prim.a_grid_lin, dist_egm', label = [L"z_1" L"z_2"], dpi=300, title = "Probability of Assets by Productivity EGM")
xaxis!("Assets")
yaxis!("Probability")
savefig(dist_egm_plot,"dist_egm.png")

ergodic_dist_egm = plot(prim.a_grid_lin, dist_egm'.*kpol', label = [L"z_1" L"z_2"], dpi=300, title = "Distribution of Assets by Productivity VFI")
xaxis!("Assets")
yaxis!("Distribution")
savefig(ergodic_dist_egm,"ergodic_dist_egm.png")

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


function parameter_setup()
    
    nd = 100
    dmax = 40
    dmin = 0
    d_grid = exp.( range( log(1+dmin), log(1+dmax), nd)  ) .-1  # will be created later again same for tauchen...

    ρ = 0.95
    nz = 5
    mc = QuantEcon.tauchen(nz, ρ, 4, 0, 3)
    z_logs = mc.state_values
    trans_mat = mc.p

    #z_grid = zeros(nz)
    #for i = 1:nz
    #    z_grid[i] = exp(z_logs[i] - 0.5*(4^2/(1-ρ^2)))
    #end
    z_grid = [0.08964770, 0.38482210, 0.73655050, 1.2183520, 2.57389100000000]
    γ = 0.95
    y_gam = γ*minimum(z_grid)

    
    #y_gam = 0.085


    # capital grid
    μ, δ = 0.97, 0.02
    xmin, xmax = -y_gam + (1 - μ)*(1 - δ)*dmin, 60.
    nx = 225
    x_grid = exp.( exp.(exp.( range(0, log( log( log(xmax - xmin + 1) +1) +1), nx)  ) .-1 ) .-1) .-1 .+ xmin
    #k_grid =  range(  kmin , kmax ,nk)

    pm = (
        β = 0.9391,       # discount factor for comparability
        σ = 2.00,       # CRRA (inverse = IES constant) 
        γ = 0.95,       # utility loss of working 
        r = 0.03,
        θ = 0.8092,
        δ = 0.02,
        μ = 0.97,
        ϵ_dur = 0.000001,

        tol = 0.00001,
        maxit_pol = 500,

        xmin = xmin,
        xmax = xmax,
        nx = nx,
        x_grid = x_grid,

        y_gam = y_gam,

        dmin = dmin,
        dmax = dmax,
        nd = nd,
        d_grid = exp.(exp.(exp.(range(0,log(log(log(dmax - dmin+1)+1)+1),nd)).-1).-1).-1 .+dmin,
        
        nz = nz,
        z_grid = [0.08964770, 0.38482210, 0.73655050, 1.2183520, 2.57389100000000],
        Pr_z = [0.98538234869983 0.014617648428214 2.87194912296229e-09 2.66453525910038e-15 0;
        0.00454314813291247	0.845112816239605	0.149116620059576 0.00122731240726603 1.03160640585465e-07;
        1.05379593784682e-06 0.135866597717910	0.678690722832698	0.184333151316438 0.00110847433701633;
        7.91306385469355e-11 0.00294861102200996	0.220814171194626	0.696248765541488 0.0799884521627456;
        4.15627872233493e-19 9.34090430312097e-08	0.000645115643546287	0.145432208996651 0.853922581950759], 
    )

end

function d_prime(pm, v_hat_xprime, v_hat_dprime)

    # DISREGARD THIS FUNCTION - ALL RELEVANT PARTS IN OTHER FUNCTION

    @unpack nz, nx, z_grid, d_grid, x_grid, dmin, r, δ, dmin = pm

    diff_Der = v_hat_dprime - (r + δ).*v_hat_xprime # Wealth at expectation
    d_prime_opt = zeros(nx, nz) # storage optimal d_prime
    kappa_xy = similar(d_prime_opt) # Storage of Lagrange Multiplier
    v_hat_xprimeopt= similar(d_prime_opt) # Storing of derivative at optimal condition
    
    for (z_idx, _) in enumerate(z_grid)
        println("Shock #: ", z_idx)
        for (xp_idx, x_next) in enumerate(x_grid)
            #itp = interpolate((diff_Der[:, 1, 1],), d_grid, Gridded(Linear()))
            println("Capital #: ", xp_idx)
            d_prime_now = extrapolate( interpolate( (diff_Der[:, xp_idx, z_idx],), d_grid, Gridded(Linear())), Interpolations.Flat() )(0)

            # Here, we have an interior solution
            if (d_prime_now > dmin) && (d_prime_now < (y_gam + x_next)/((1 - μ)*(1 - δ)) )
                d_prime_opt[xp_idx, z_idx] = d_prime_now
                v_hat_xprimeopt[xp_idx, z_idx] = v_hat_xprime[xp_idx, z_idx]

            # Corner Solution at non-negativity constraint
            elseif diff_Der[1, xp_idx, z_idx] <= 0
            # Why only first d' element? Can we say that when it doesn't apply here in no other neither? What about if it applies?
                d_prime_opt[xp_idx, z_idx] = dmin
                
            else # If collateral constraint binds
                d_prime_opt[xp_idx, z_idx] = (y_gam + x_now)/((1 - μ)*(1 - δ))
                v_hat_xprimeopt[xp_idx, z_idx] = v_hat_xprime[xp_idx, z_idx]
                v_hat_xprimeopt[xp_idx, z_idx] = extrapolate( interpolate( (d_grid, ), log.(v_hat_xprime[:, xp_idx, z_idx]) , Gridded(Linear())), Interpolations.Flat() )(d_prime_opt[xp_idx, z_idx])
                v_hat_dprimeopt = extrapolate( interpolate( (d_grid, ), log.(v_hat_dprime[:, xp_idx, z_idx]) , Gridded(Linear())), Interpolations.Flat() )(d_prime_opt[xp_idx, z_idx])
                kappa_xy[xp_idx, z_idx] =  (1/(1 + r - μ*(1 - δ)))*(v_hat_dprimeopt - (r + δ)*v_hat_xprimeopt[xp_idx, z_idx])
            end
        end
    end

    return
end


function EGM(pm)

    @unpack θ, ϵ_dur, σ, nd, nx, nz, d_grid, x_grid, Pr_z, β, z_grid, r, δ, dmin, μ, y_gam = pm
    
    c_pol_guess = zeros(nd, nx, nz)
    for (z_idx, z) in enumerate(z_grid)
        for (d_idx, _) in enumerate(d_grid)
            c_pol_guess[d_idx,:,z_idx] = x_grid .+ z
        end
    end

    iter = 0
    maxiter = 200
    tol = 1e-6
    err = 1

    while (err > tol) && (iter < maxiter)
        
        a_prime = similar(c_pol_guess)
        d_prime = similar(c_pol_guess)
        c_prime = similar(c_pol_guess)

        MUc = zeros(nd, nx, nz)
        MUd = zeros(nd, nx, nz)
        v_hat_xprime = zeros(nd, nx, nz)
        v_hat_dprime = zeros(nd, nx, nz)
        diff_Der = zeros(nd, nx, nz)

        d_prime_x = zeros(nx, nz) # storing optimal d'
        v_hat_xprimeopt = zeros(nx, nz) # storing derivative
        v_hat_dprimeopt = zeros(nx, nz)
        kappa_xy = zeros(nx, nz) # lagrange multiplier storage

        # Derivative of Marginal Utilities
        for (z_idx, _) in enumerate(z_grid)
            println("Productivity State: ", z_idx)

            for (x_idx, _) in enumerate(x_grid)
                MUc[:, x_idx, z_idx] = θ .* (c_pol_guess[:, 1, 1].^θ .*(d_grid .+ ϵ_dur).^(1-θ)).^(-σ) .* (d_grid .+ ϵ_dur).^(1-θ) .* c_pol_guess[:, x_idx, z_idx].^(θ-1)
                MUd[:, x_idx, z_idx] = (1-θ) .* (c_pol_guess[:, x_idx, z_idx].^θ.*(d_grid .+ ϵ_dur).^(1-θ)).^(-σ) .* (d_grid .+ ϵ_dur).^( -θ) .* c_pol_guess[:, x_idx, z_idx].^(θ)

            println("==> Marginal Utilities calculated")


            # Expected Values at the margin
                v_hat_xprime[:, x_idx, z_idx] = β .* MUc[:, x_idx, :] * Pr_z[z_idx, :]
                v_hat_dprime[:, x_idx, z_idx] = β .* MUd[:, x_idx, :] * Pr_z[z_idx, :] # damn, they are still off :( but better
            end
            println("==> Expected Values of each Asset calculated")
        end

        # If it's an interior solution, this is the outcome.
        diff_Der[:,:, :] = v_hat_dprime - (r + δ).*v_hat_xprime

        for (z_idx, _) in enumerate(z_grid)
        # Let's check if that's actually true
            for (xp_idx, x_prime) in enumerate(x_grid)
                println("Capital Tomorrow: ", xp_idx)

                # Sort diff_Der and keep track of the index
                current_foc = diff_Der[:, xp_idx, z_idx]
                order = sortperm(diff_Der[:, xp_idx, z_idx], rev=false)

                # For given future wealth(assets), find the optimal d' that satisfies Budget Constraint directly with zero (include expected values)
                d_prime_now = extrapolate( interpolate( (current_foc[order],), d_grid[order], Gridded(Linear())), Interpolations.Flat() )(0)
                (xp_idx % 25  == 0) ? println("Optimal Durable As Response: ", d_prime_now) : nothing

                # If Interior Solution, interpolated on grid and picked point 0
                if  (d_prime_now > dmin) && (d_prime_now < (x_prime + y_gam)/((1 - μ)*(1 - δ)))
                    println("INTERIOR")
                    d_prime_x[xp_idx, z_idx] = d_prime_now
                    v_hat_xprimeopt[xp_idx, z_idx] = extrapolate( interpolate( (d_grid,), log.(v_hat_xprime[:, xp_idx, z_idx]), Gridded(Linear())), Interpolations.Flat() )(d_prime_x[xp_idx, z_idx])
                elseif d_prime_now <= 0
                    println("CORNER NON-NEGATIVITY")
                    d_prime_x[xp_idx, z_idx] = dmin # Here, we impose (economically) - if optimal d_prime is lower than 0 - the lowest value possible (economically/on the grid)
                    v_hat_xprimeopt[xp_idx, z_idx] = v_hat_xprime[1, xp_idx, z_idx] # No interpolation necessary
                else
                    println("CORNER COLLATERAL")
                    d_prime_x[xp_idx, z_idx] = (x_prime + y_gam)/((1 - μ)*(1 - δ)) # Impose that we can't get more durables than this
                    # Get optimal derivatives
                    v_hat_xprimeopt[xp_idx, z_idx] = extrapolate( interpolate( (d_grid, ), log.(v_hat_xprime[:, xp_idx, z_idx]) , Gridded(Linear())), Interpolations.Flat() )(d_prime_x[xp_idx, z_idx])
                    v_hat_dprimeopt[xp_idx, z_idx] = extrapolate( interpolate( (d_grid, ), log.(v_hat_dprime[:, xp_idx, z_idx]) , Gridded(Linear())), Interpolations.Flat() )(d_prime_x[xp_idx, z_idx])
                    # See where lagrange multiplier binds
                    kappa_xy[xp_idx, z_idx] =  (1/(1 + r - μ*(1 - δ)))*(v_hat_dprimeopt[xp_idx, z_idx] - (r + δ)*v_hat_xprimeopt[xp_idx, z_idx])
                end
            end
        end
        # End of step 1)
        Plots.plot(x_grid, d_prime_x, title="next period combinations, d´(x´)")

        # Step 2 
        for (z_idx, _) in enumerate(z_grid)
            for (d_idx, d_now) in enumerate(d_grid) # Given todays durables

                # The next three equation are endogenous grid points. We need to interpolate and extrapolate later.

                # Equation (13) - Optimality condition for consumption
                c_egm = ((1 + r)/θ .* (v_hat_xprimeopt[:, z_idx] + kappa_xy[:, z_idx]).*(d_now + ϵ_dur).^((θ - 1)*(1-σ))).^(1/(θ*(1 - σ) - 1))
                # Equation (1) - Definition of assets in terms of total wealth and durables
                a_prime_EGM = (x_grid .- (1 - δ).*d_prime_x[:,z_idx])./(1 + r)
                # Back out total wealth from Budget Constraint
                x_EGM = c_EGM + a_prime_EGM + dprime_xy[:,z_idx] - z_grid[z_idx]

                # where capital on the grid < capital of endogenous grid, we want to have as much (non-durable) consumption as possible
                # therefore, we know that consume total wealth (both collateral constraint and dprime=d_min are binding)
                c_pol_new[d_idx, findall(x_grid .< minimum(x_EGM)), z_idx] .= min(c_EGM) .- (minimum(x_EGM) .- x_grid[findall(x_grid .< minimum(x_EGM))])

                # Otherwise, we have to interpolate
                c_pol_new[d_idx, findall(x_grid .>= minimum(x_EGM)),z_idx] .=  extrapolate( interpolate( (x_EGM, ), c_EGM, Gridded(Linear())), Interpolations.Flat() )(findall(x_grid .>= minimum(x_EGM)))
                
                # This here, I don't understand. Why do we extrapolate on the x_grid?????
                d_prime[d_idx,:,z_idx] .= maximum(extrapolate( interpolate( (x_EGM, ), d_prime_x[:,z_idx], Gridded(Linear()) ), Interpolations.Flat())(x_grid))
                # Backout future assets as grid points from Budget constraint
                a_prime[d_idx,:,z_idx] .= x_grid .+ z_grid[z_idx] .- c_pol_new[d_idx,:,z_idx] .- d_prime[d_idx,:,z_idx]
                # Backout grid point wealth based on wealth definition Eq. (1)
                x_prime[d_idx,:,z_idx] .= (1+r)*a_prime[d_idx,:,z_idx] + (1-δ)*d_prime[d_idx,:,z_idx]

            end
        end

        err  = norm( c_pol_guess .- c_pol_new) / (1 + norm(c_pol_guess) )
        c_pol_guess   = deepcopy(c_pol_new)

        println("Iteration: ", iter, "with error: ", err)

    end 
    
    # Read what Robert sent (Appendix)
    # Auclert Appendix e) egm
    

    return 
end

pm = parameter_setup()

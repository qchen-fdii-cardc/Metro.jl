using Metro
using Test
using Random
using Statistics

@testset "Metro.jl" begin
    # MATLAB-like helper: normalized histogram density estimate.
    function histnorm(x::AbstractVector{<:Real}, nbins::Integer)
        @assert nbins > 0
        lo, hi = extrema(x)
        if hi == lo
            hi = lo + 1.0
        end
        edges = collect(range(lo, hi; length = nbins + 1))
        widths = diff(edges)
        counts = zeros(Int, nbins)
        for xi in x
            b = searchsortedlast(edges, xi)
            b = clamp(b, 1, nbins)
            counts[b] += 1
        end
        density = counts ./ (length(x) .* widths)
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        return centers, density, edges
    end

    # MATLAB-like helper: Gaussian KDE with Silverman's bandwidth rule.
    function kde_gaussian(x::AbstractVector{<:Real}; ngrid::Integer = 256)
        n = length(x)
        @assert n > 1
        s = std(x)
        h = max(1.06 * s * n^(-1 / 5), eps(Float64))
        lo, hi = extrema(x)
        grid = collect(range(lo - 3h, hi + 3h; length = ngrid))

        invsqrt2π = 1 / sqrt(2π)
        density = similar(grid)
        for (i, xi) in pairs(grid)
            z = (xi .- x) ./ h
            density[i] = mean(invsqrt2π .* exp.(-0.5 .* z .^ 2)) / h
        end

        # Trapezoidal integration for CDF and normalization checks.
        cdf = zeros(Float64, ngrid)
        for i in 2:ngrid
            dx = grid[i] - grid[i - 1]
            cdf[i] = cdf[i - 1] + 0.5 * (density[i - 1] + density[i]) * dx
        end

        return h, density, grid, cdf
    end

    # Equivalent to MATLAB's normpdf for vectors.
    normpdf(x, μ, σ) = @. exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

    @testset "height_example MATLAB parity" begin
        Random.seed!(42)

        n = 400
        μ = 67.0
        σ = 2.5

        heights = μ .+ σ .* randn(n)
        students = 1:n

        @test length(heights) == n
        @test first(students) == 1
        @test last(students) == n

        # Histogram + KDE + target Gaussian density on +/- 3 sigma grid.
        xgrid = collect(range(μ - 3σ, μ + 3σ; length = 1001))
        pdf = normpdf(xgrid, μ, σ)

        bandwidth, density_heights, heights_mesh, cdf_heights = kde_gaussian(heights)
        centers, hist_density, edges = histnorm(heights, 15)

        @test bandwidth > 0
        @test length(density_heights) == length(heights_mesh)
        @test length(cdf_heights) == length(heights_mesh)
        @test all(>=(0), density_heights)
        @test all(>=(0), hist_density)
        @test all(diff(cdf_heights) .>= -1e-12)

        # Numerical integration checks for normalized histogram and KDE.
        hist_mass = sum(hist_density .* diff(edges))
        kde_mass = cdf_heights[end]
        @test isapprox(hist_mass, 1.0; atol = 0.08)
        @test isapprox(kde_mass, 1.0; atol = 0.08)

        dxgrid = xgrid[2] - xgrid[1]
        @test isapprox(sum(pdf) * dxgrid, 1.0; atol = 0.01)

        # Sample mean/std, mirroring MATLAB xbar, S^2, and S.
        xbar = mean(heights)
        S2 = sum((heights .- xbar) .^ 2) / (n - 1)
        S = sqrt(S2)

        @test isapprox(xbar, μ; atol = 0.35)
        @test isapprox(S, σ; atol = 0.30)

        # 95% CI half-width using known sigma (MATLAB: 2*sigma/sqrt(n)).
        value_known_sigma = 2 * σ / sqrt(n)
        value_unknown_sigma_large_n = 1.96 * S / sqrt(n)
        @test value_known_sigma > 0
        @test value_unknown_sigma_large_n > 0
        @test isapprox(value_unknown_sigma_large_n, value_known_sigma; rtol = 0.15)

        # Sampling distributions of xbar for N = 40, 400, 4000.
        Ns = (40, 400, 4000)
        pdf_peaks = Float64[]
        pdf_vars = Float64[]
        for N in Ns
            σx = σ / sqrt(N)
            xN = collect(range(μ - 3σx, μ + 3σx; length = 1001))
            pdfN = normpdf(xN, μ, σx)
            dxN = xN[2] - xN[1]
            push!(pdf_peaks, maximum(pdfN))
            push!(pdf_vars, σx^2)
            @test isapprox(sum(pdfN) * dxN, 1.0; atol = 0.01)
        end

        @test pdf_vars[1] > pdf_vars[2] > pdf_vars[3]
        @test pdf_peaks[1] < pdf_peaks[2] < pdf_peaks[3]
    end
end

module FeatureUtilities
using Features
using Test
using LinearRegression, Statistics

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1
Features.verbosity = 1


function normalize_y(y, regr, grad, yix )
    normy = similar(y)
    for ix in eachindex(y)
        # normy[ix] = y[yix] - (regr - grad * (yix - ix))
        normy[ix] = (regr - grad * (yix - ix))
    end
    return normy
end

medvol(v, w) = [median(v[(max(firstindex(v),ix-w+1):ix)]) for ix in eachindex(v)]

@testset "relativevolume tests" begin
    vol = collect(0f0:6f0)
    sl = 2
    ll = 3
    res = Features.relativevolume(vol, sl, ll)
    rvtest = vcat([vol[1] / eps(Float32)], (medvol(vol, sl) ./ medvol(vol, ll))[2:end])
    @test all(rvtest .== res)
    
end

@testset "rollingmax tests" begin

    res = Features.rollingmax(collect(2:11), 3)
    @test all(collect(2:11) .== res)
    res = Features.rollingmax(collect(11:-1:2), 3)
    @test all(vcat([11, 11], collect(11:-1:4)) .== res)
    
end

@testset "rollingmin tests" begin

    res = Features.rollingmin(collect(2:11), 3)
    @test all(vcat([2, 2], collect(2:9)) .== res)
    res = Features.rollingmin(collect(11:-1:2), 3)
    @test all(collect(11:-1:2) .== res)
    
end

@testset "rollingregression tests" begin

y = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
for (window, startix) in [(8,3), (4,3), (4,1), (4,6)]  #TODO 
    (verbosity >= 2) && println("\nwindow=$window, startix=$startix")
    r, g = Features.rollingregression(y, window, startix)
    s, m,n = Features.rollingregressionstd(y, r, g, window, startix)
    smv = Features.rollingregressionstdmv([y], r, g, window, startix)
    @test all(isapprox.(smv, s, atol=eps(Float32)))
    smv2 = Features.rollingregressionstdmv([y, y], r, g, window, startix)
    # println("smv=$smv != s=$s")
    # @test all(isapprox.(smv, s, atol=eps(Float32)))
    (verbosity >= 2) && println("r=$r, g=$g")
    (verbosity >= 2) && println("len(s, m, n)=$(length(s)), $(length(m)), $(length(n)),  s=$s, m=$m, n=$n")
    lrnorm = similar(n)
    lrmean = similar(m)
    lrstd = similar(s)
    lrstd2 = similar(s)
    for ix in startix:length(y)
        win = ix < window ? ix : window
        relix = ix-win+1
        yv = view(y, relix:ix)
        X = collect(-win+1:0)
        (verbosity >= 3) && println("win=$win, yv=y[relix=$relix:ix=$ix]=$yv, X=$X")
        lr = linregress(X, yv)
        lrslope = LinearRegression.slope(lr)[1]
        lrbias = LinearRegression.bias(lr)
        (verbosity >= 3) && println("ix=$ix, test=$((lrslope ≈ g[ix-startix+1]) && (lrbias ≈ r[ix-startix+1])), X=$X, yv=$yv, relix=$relix:ix=$ix, lrslope=$lrslope ≈ g[ix-startix+1=$(ix-startix+1)]=$(g[ix-startix+1]) = $(lrslope ≈ g[ix-startix+1]), lrbias=$lrbias ≈ r[ix-startix+1]=$(r[ix-startix+1]) = $(lrbias ≈ r[ix-startix+1])")
        @test lrbias ≈ r[ix-startix+1]
        @test lrslope ≈ g[ix-startix+1]
        ny = normalize_y(y, r[ix-startix+1], g[ix-startix+1], ix )
        offset = max(ix, win)
        # lrny = y .- [lr([lrix]) for lrix in -length(yv)+1:length(y)-length(yv)]
        lrny = [lr([lrix]) for lrix in -offset+1:-offset+length(y)]
        (verbosity >= 3) && println("ny=$ny, lrny=$lrny, test=[$([ny[i] ≈ lrny[i] for i in eachindex(ny)])] testall=$(all([ny[i] ≈ lrny[i] for i in eachindex(ny)]))")
        @test all(ny .≈ lrny)
        lrs = yv - lrny[relix:ix]
        (verbosity >= 3) && println("lrs=$lrs")
        lrnorm[ix-startix+1] = lrs[end]
        lrmean[ix-startix+1] = Statistics.mean(lrs)
        length(lrs) > 1 ? lrstd[ix-startix+1] = Statistics.stdm(lrs, lrmean[ix-startix+1]) : 0
        length(lrs) > 1 ? lrstd2[ix-startix+1] = Statistics.stdm(vcat(lrs, lrs), Statistics.mean(vcat(lrs, lrs))) : 0
    end
    (verbosity >= 2) && println("len(lrstd, lrmean, lrnorm)=$(length(lrstd)), $(length(lrmean)), $(length(lrnorm)),   lrstd=$lrstd, lrmean=$lrmean, lrnorm=$lrnorm")
    @test all(isapprox.(lrstd2, smv2, atol=eps(Float32)))
    @test all(isapprox.(lrstd, s, atol=eps(Float32)))
    @test all(isapprox.(lrmean, m, atol=eps(Float32)))
    @test all(isapprox.(lrnorm, n, atol=eps(Float32)))
    # (verbosity >= 2) && println("std_test=$(all(s .≈ lrstd)), lrs=$lrs, lrmean=$lrmean, lrstd=$lrstd")
end

end # test set
end # module
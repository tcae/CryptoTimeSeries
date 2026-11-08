"""
Centered regression statistics for consecutive, equidistant samples.
`covariance` is the sum of centered sample index times centered price.
"""
struct RegressionMoments
    count::Int
    mean::Float64
    covariance::Float64
end

"""Return empty equidistant regression statistics."""
RegressionMoments() = RegressionMoments(0, 0.0, 0.0)

"""Append a price using centered arithmetic without subtracting large raw sums."""
function appendprice(stats::RegressionMoments, price::Real)::RegressionMoments
    count = stats.count + 1
    delta = price - stats.mean
    return RegressionMoments(count, stats.mean + delta / count,
        stats.covariance + stats.count * delta / 2)
end

"""Remove the first price and rebase the remaining equidistant sample coordinates."""
function removeprice(stats::RegressionMoments, price::Real)::RegressionMoments
    stats.count == 1 && return RegressionMoments()
    delta = stats.mean - price
    return RegressionMoments(stats.count - 1, stats.mean + delta / (stats.count - 1),
        stats.covariance - stats.count * delta / 2)
end

"""Return the fitted endpoint and slope per sample from centered statistics."""
function regressionresult(stats::RegressionMoments)
    @assert stats.count > 0 "regression requires samples; count=$(stats.count)"
    slope = stats.count == 1 ? 0.0 :
        stats.covariance / (stats.count * (stats.count - 1.0) * (stats.count + 1.0) / 12)
    return (regry=stats.mean + slope * (stats.count - 1) / 2, grad=slope)
end

"""Calculate centered regression statistics for an inclusive price segment."""
function regressionmoments(prices::AbstractVector, firstix::Int, lastix::Int)::RegressionMoments
    stats = RegressionMoments()
    for ix in firstix:lastix
        stats = appendprice(stats, prices[ix])
    end
    return stats
end

"""Merge adjacent equidistant segments using centered mean differences."""
function mergeregression(left::RegressionMoments, right::RegressionMoments)::RegressionMoments
    left.count == 0 && return right
    right.count == 0 && return left
    count = left.count + right.count
    delta = right.mean - left.mean
    return RegressionMoments(count, left.mean + delta * right.count / count,
        left.covariance + right.covariance + (left.count / 2) * right.count * delta)
end

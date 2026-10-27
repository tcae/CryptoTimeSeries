"""SimpleChains model with owned parameters, layer state and reusable gradient storage."""
struct SimpleChainsModel{C}
    chain::C
    parameters::Vector{Float32}
    gradient::Vector{Float32}
    outputcount::Int
end

"""Evaluate logits, returning owned storage rather than SimpleChains scratch memory."""
function (model::SimpleChainsModel)(features::AbstractMatrix{Float32})
    logits = Matrix{Float32}(undef, model.outputcount, size(features, 2))
    logits .= model.chain(features, model.parameters)
    return logits
end

"""Set native BatchNorm training state; stateless dense layers need no mode change."""
_simplechainsmode!(::SimpleChains.TurboDense, ::Bool) = nothing

"""Switch native BatchNorm between batch statistics and frozen running statistics."""
_simplechainsmode!(layer::SimpleChainsBatchNorm, training::Bool) = (layer.training = training; nothing)

"""Enable training behavior for every native stateful layer."""
function Flux.trainmode!(model::SimpleChainsModel)
    foreach(layer -> _simplechainsmode!(layer, true), model.chain.layers)
    return model
end

"""Freeze running statistics for deterministic inference."""
function Flux.testmode!(model::SimpleChainsModel)
    foreach(layer -> _simplechainsmode!(layer, false), model.chain.layers)
    return model
end

"""
    model011(featurecount, labels, mnemonic)::NN

Architecture-matched native SimpleChains implementation of model008, including
all four affine BatchNorm layers, Flux-compatible moving statistics, identical
seeded initialization, and persistent Adam(0.0001, (0.9, 0.999)).
Training batches must contain at least two observations.
"""
function model011(featurecount::Int, labels::AbstractVector, mnemonic::AbstractString)::NN
    @assert featurecount > 0 "featurecount=$(featurecount) must be positive"
    @assert !isempty(labels) "labels=$(labels) must not be empty"
    reference = model008(featurecount, labels, mnemonic)
    layers = map(reference.model.layers) do layer
        if layer isa Flux.BatchNorm
            SimpleChainsBatchNorm(length(layer.β); eps=layer.ϵ, momentum=layer.momentum)
        else
            SimpleChains.TurboDense(layer.σ, length(layer.bias))
        end
    end
    parameters = reduce(vcat, map(reference.model.layers) do layer
        layer isa Flux.BatchNorm ? vcat(layer.β, layer.γ) : vcat(vec(layer.weight), layer.bias)
    end)
    chain = SimpleChains.SimpleChain(SimpleChains.static(featurecount), layers...)
    model = SimpleChainsModel(chain, parameters, similar(parameters), length(labels))
    nn = NN(model, Flux.setup(Flux.Adam(0.0001, (0.9, 0.999)), parameters),
        Flux.logitcrossentropy, labels, "Native SimpleChains: " * reference.description)
    setmnemonic(nn, "model011_" * mnemonic)
    return nn
end

"""
Weighted mean logit cross entropy for SimpleChains, using the existing one-hot
batch contract. Its pullback is normalized by the sum of sample weights.
"""
struct SimpleChainsWeightedLoss{Y,W} <: SimpleChains.AbstractLoss{Y}
    y::Y
    weights::W
end

"""Return the target buffer retained during SimpleChains evaluation."""
SimpleChains.target(loss::SimpleChainsWeightedLoss) = loss.y

"""Calculate stable weighted cross entropy and optionally replace logits by their gradient."""
function _simplechainsloss!(logits::AbstractMatrix, loss::SimpleChainsWeightedLoss, gradient::Bool)
    weightsum = isnothing(loss.weights) ? size(logits, 2) : sum(loss.weights)
    @assert weightsum > 0 "sum of sample weights=$(weightsum) must be positive"
    total = 0f0
    for sample in axes(logits, 2)
        maxlogit = maximum(view(logits, :, sample))
        expsum = sum(value -> exp(value - maxlogit), view(logits, :, sample))
        logdenom = log(expsum)
        weight = isnothing(loss.weights) ? 1f0 : loss.weights[sample]
        scale = weight / weightsum
        for label in axes(logits, 1)
            shifted = logits[label, sample] - maxlogit
            target = loss.y[label, sample]
            total -= scale * target * (shifted - logdenom)
            if gradient
                logits[label, sample] = scale * (exp(shifted) / expsum - target)
            end
        end
    end
    return total
end

"""SimpleChains terminal loss pullback; gradients already include mean-loss scaling."""
function SimpleChains.chain_valgrad!(_, logits::AbstractMatrix,
    layers::Tuple{SimpleChainsWeightedLoss{Y,W}}, parameters::Ptr, memory::Ptr{UInt8}) where {Y,W}
    return _simplechainsloss!(logits, only(layers), true), logits, memory
end

"""Evaluate the terminal weighted loss without overwriting logits."""
function (loss::SimpleChainsWeightedLoss)(logits::AbstractMatrix, parameters::Ptr, memory::Ptr{UInt8})
    return _simplechainsloss!(logits, loss, false), parameters, memory
end

"""
    model009(featurecount, labels, mnemonic)::NN

SimpleChains dense-only alternative to model008, with the same hidden widths,
ReLU activations and Adam(0.0001, (0.9, 0.999)), but no BatchNorm. This is an
architectural alternative, not a numerically equivalent replacement for model008.
"""
function model009(featurecount::Integer, labels::AbstractVector, mnemonic::AbstractString)::NN
    @assert featurecount > 0 "featurecount=$(featurecount) must be positive"
    @assert !isempty(labels) "labels=$(labels) must not be empty"
    control = model010(featurecount, labels, mnemonic)
    layers = control.model.layers
    chain = SimpleChains.SimpleChain(SimpleChains.static(featurecount),
        (SimpleChains.TurboDense(layer.σ, length(layer.bias)) for layer in layers)...)
    parameters = Vector{Float32}(undef, sum(layer -> length(layer.weight) + length(layer.bias), layers))
    offset = 0
    for layer in layers
        nweights = length(layer.weight)
        parameters[offset+1:offset+nweights] .= vec(layer.weight)
        offset += nweights
        parameters[offset+1:offset+length(layer.bias)] .= layer.bias
        offset += length(layer.bias)
    end
    model = SimpleChainsModel(chain, parameters, similar(parameters), length(labels))
    nn = NN(model, Flux.setup(Flux.Adam(0.0001, (0.9, 0.999)), parameters),
        Flux.logitcrossentropy, labels, "SimpleChains dense-only alternative: " * control.description)
    setmnemonic(nn, "model009_" * mnemonic)
    return nn
end

"""
    model010(featurecount, labels, mnemonic)::NN

Flux dense-only control for model009: model008's widths, activations and optimizer
with all BatchNorm layers removed. With the same Random seed, model009 and
model010 start with identical dense weights and biases.
"""
function model010(featurecount::Integer, labels::AbstractVector, mnemonic::AbstractString)::NN
    @assert featurecount > 0 "featurecount=$(featurecount) must be positive"
    @assert !isempty(labels) "labels=$(labels) must not be empty"
    firstwidth = 3 * featurecount
    secondwidth = round(Int, firstwidth * 2 / 3)
    thirdwidth = round(Int, (secondwidth + length(labels)) / 2)
    model = Flux.Chain(Flux.Dense(featurecount => firstwidth, Flux.relu),
        Flux.Dense(firstwidth => secondwidth, Flux.relu),
        Flux.Dense(secondwidth => thirdwidth, Flux.relu),
        Flux.Dense(thirdwidth => length(labels)))
    nn = NN(model, Flux.setup(Flux.Adam(0.0001, (0.9, 0.999)), model),
        Flux.logitcrossentropy, labels, "Dense-only $(featurecount)-$(firstwidth)-$(secondwidth)-$(thirdwidth)-$(length(labels))")
    setmnemonic(nn, "model010_" * mnemonic)
    return nn
end

"""Train one batch of a Flux model using its configured weighted or unweighted loss."""
function trainbatch!(nn::NN, features::AbstractMatrix{Float32}, targets::AbstractMatrix,
    weights::Union{Nothing,AbstractVector{Float32}})
    return _trainbatch!(nn.model, nn, features, targets, weights)
end

"""Differentiate and update a Flux model for one batch."""
function _trainbatch!(model, nn::NN, features, targets, weights)
    loss, gradients = Flux.withgradient(model) do current
        logits = nn.valuecorrection(current(features))
        isnothing(weights) ? nn.lossfunc(logits, targets) : weightedloss(nn, logits, targets, weights)
    end
    Flux.update!(nn.optim, model, gradients[1])
    return loss
end

"""Differentiate a SimpleChains batch and update with the same persistent Adam as Flux."""
function _trainbatch!(model::SimpleChainsModel, nn::NN, features, targets, weights)
    @assert nn.lossfunc === Flux.logitcrossentropy "SimpleChains requires logitcrossentropy, got $(nn.lossfunc)"
    @assert size(features, 2) == size(targets, 2) "batch sizes differ: features=$(size(features)), targets=$(size(targets))"
    @assert size(targets, 1) == model.outputcount "target rows=$(size(targets, 1)) must equal output count=$(model.outputcount)"
    @assert isnothing(weights) || length(weights) == size(features, 2) "weights length=$(length(weights)) differs from batch size=$(size(features, 2))"
    loss = SimpleChains.valgrad!(model.gradient,
        SimpleChains.add_loss(model.chain, SimpleChainsWeightedLoss(targets, weights)),
        features, model.parameters)
    Flux.update!(nn.optim, model.parameters, model.gradient)
    return loss
end

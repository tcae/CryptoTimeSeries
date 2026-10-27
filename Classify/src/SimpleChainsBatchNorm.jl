"""
    SimpleChainsBatchNorm(channels; eps=1f-5, momentum=0.1f0)

Native SimpleChains BatchNorm for Float32 channel-by-observation matrices.
Parameters in the chain's flat vector are `[beta; gamma]`; moving statistics
are nontrainable layer state. Training normalizes with population variance and
updates moving variance with Bessel correction, matching Flux BatchNorm.
Training requires at least two observations; singleton inference is supported.
"""
mutable struct SimpleChainsBatchNorm
    mean::Vector{Float32}
    variance::Vector{Float32}
    epsilon::Float32
    momentum::Float32
    training::Bool
end

"""Construct an affine BatchNorm layer with tracked statistics and inference mode initially."""
function SimpleChainsBatchNorm(channels::Int; eps::Float32=1f-5, momentum::Float32=0.1f0)
    @assert channels > 0 "channels=$(channels) must be positive"
    @assert eps > 0f0 "eps=$(eps) must be positive"
    @assert 0f0 <= momentum <= 1f0 "momentum=$(momentum) must be within [0, 1]"
    return SimpleChainsBatchNorm(zeros(Float32, channels), ones(Float32, channels), eps, momentum, false)
end

"""BatchNorm has a trainable offset and scale for every channel."""
SimpleChains.parameter_free(::SimpleChainsBatchNorm) = false

"""Return parameter count and unchanged output dimensions, validating the channel count."""
function SimpleChains.numparam(layer::SimpleChainsBatchNorm, dimensions::Tuple)
    @assert first(dimensions) == length(layer.mean) "input channels=$(first(dimensions)) must equal BatchNorm channels=$(length(layer.mean))"
    return 2 * length(layer.mean), dimensions
end

"""Initialize affine offsets to zero and scales to one in the chain parameter buffer."""
function SimpleChains.init_params!(layer::SimpleChainsBatchNorm, parameters::Ptr{Float32},
    dimensions::Tuple, ::Random.AbstractRNG)
    count, outputdimensions = SimpleChains.numparam(layer, dimensions)
    affine = SimpleChains.PtrArray(parameters, (length(layer.mean), 2))
    affine[:, 1] .= 0f0
    affine[:, 2] .= 1f0
    return parameters + count * sizeof(Float32), outputdimensions
end

"""Reserve an aligned output matrix for forward evaluation."""
function SimpleChains.forward_layer_output_size(::Val{Float32}, layer::SimpleChainsBatchNorm, dimensions::Tuple)
    SimpleChains.numparam(layer, dimensions)
    return SimpleChains.align(prod(dimensions) * sizeof(Float32)), dimensions
end

"""Reserve normalized inputs, inverse standard deviations and output for the pullback."""
function SimpleChains.layer_output_size(::Val{Float32}, layer::SimpleChainsBatchNorm, dimensions::Tuple)
    bytes, outputdimensions = SimpleChains.forward_layer_output_size(Val(Float32), layer, dimensions)
    return 2 * bytes + SimpleChains.align(length(layer.mean) * sizeof(Float32)), outputdimensions
end

"""Evaluate BatchNorm and optionally retain normalized inputs and inverse deviations."""
function _batchnormforward!(output, normalized, inversedeviation, layer::SimpleChainsBatchNorm,
    input, affine)
    channels, observations = size(input)
    @assert channels == length(layer.mean) "input channels=$(channels) must equal BatchNorm channels=$(length(layer.mean))"
    @assert observations > 0 && (!layer.training || observations > 1) "BatchNorm training=$(layer.training) requires observations>1 for unbiased moving variance; observations=$(observations)"
    @inbounds for channel in 1:channels
        average = layer.mean[channel]
        variance = layer.variance[channel]
        if layer.training
            total = 0f0
            @simd for sample in 1:observations
                total += input[channel, sample]
            end
            average = total / observations
            squared = 0f0
            @simd for sample in 1:observations
                squared += (input[channel, sample] - average)^2
            end
            variance = squared / observations
            momentum = layer.momentum
            layer.mean[channel] = (1f0 - momentum) * layer.mean[channel] + momentum * average
            layer.variance[channel] = (1f0 - momentum) * layer.variance[channel] +
                momentum * (observations / (observations - 1f0)) * variance
        end
        inverse = inv(sqrt(variance + layer.epsilon))
        scale = affine[channel, 2] * inverse
        bias = -scale * average + affine[channel, 1]
        @simd for sample in 1:observations
            output[channel, sample] = scale * input[channel, sample] + bias
        end
        if !isnothing(normalized)
            inversedeviation[channel] = inverse
            @simd for sample in 1:observations
                normalized[channel, sample] = (input[channel, sample] - average) * inverse
            end
        end
    end
    return output
end

"""Run BatchNorm directly on SimpleChains' forward scratch buffer."""
function (layer::SimpleChainsBatchNorm)(input::AbstractMatrix{Float32},
    parameters::Ptr{Float32}, memory::Ptr{UInt8})
    output = SimpleChains.PtrArray(Ptr{Float32}(memory), size(input))
    affine = SimpleChains.PtrArray(parameters, (length(layer.mean), 2))
    _batchnormforward!(output, nothing, nothing, layer, input, affine)
    return output, parameters + 2 * length(layer.mean) * sizeof(Float32),
        memory + SimpleChains.align(length(input) * sizeof(Float32))
end

"""Run the native training forward pass and retain only the data needed by its pullback."""
function SimpleChains.valgrad_layer!(gradient::Union{Nothing,Ptr{Float32}},
    layer::SimpleChainsBatchNorm, input::AbstractMatrix{Float32},
    parameters::Ptr{Float32}, memory::Ptr{UInt8})
    channels = length(layer.mean)
    matrixbytes = SimpleChains.align(length(input) * sizeof(Float32))
    normalized = SimpleChains.PtrArray(Ptr{Float32}(memory), size(input))
    inversedeviation = SimpleChains.PtrArray(Ptr{Float32}(memory + matrixbytes), (channels,))
    outputmemory = memory + matrixbytes + SimpleChains.align(channels * sizeof(Float32))
    output = SimpleChains.PtrArray(Ptr{Float32}(outputmemory), size(input))
    affine = SimpleChains.PtrArray(parameters, (channels, 2))
    _batchnormforward!(output, normalized, inversedeviation, layer, input, affine)
    nextgradient = isnothing(gradient) ? nothing : gradient + 2 * channels * sizeof(Float32)
    return nextgradient, output, parameters + 2 * channels * sizeof(Float32), outputmemory + matrixbytes
end

"""Reduce the incoming gradient to affine offset and scale gradients."""
function SimpleChains.pullback_param!(gradient::Ptr{Float32}, layer::SimpleChainsBatchNorm,
    incoming, input, ::Ptr{Float32}, memory::Ptr{UInt8})
    channels, observations = size(input)
    normalized = SimpleChains.PtrArray(Ptr{Float32}(memory), size(input))
    affinegradient = SimpleChains.PtrArray(gradient, (channels, 2))
    @inbounds for channel in 1:channels
        offsetgradient = scalegradient = 0f0
        @simd for sample in 1:observations
            value = incoming[channel, sample]
            offsetgradient += value
            scalegradient += value * normalized[channel, sample]
        end
        affinegradient[channel, 1] = offsetgradient
        affinegradient[channel, 2] = scalegradient
    end
    return nothing
end

"""Apply the BatchNorm input Jacobian without materializing a Jacobian matrix."""
function _batchnormpullback!(output, incoming, layer::SimpleChainsBatchNorm, input,
    parameters::Ptr{Float32}, memory::Ptr{UInt8})
    channels, observations = size(input)
    normalized = SimpleChains.PtrArray(Ptr{Float32}(memory), size(input))
    inversedeviation = SimpleChains.PtrArray(
        Ptr{Float32}(memory + SimpleChains.align(length(input) * sizeof(Float32))), (channels,))
    affine = SimpleChains.PtrArray(parameters, (channels, 2))
    @inbounds for channel in 1:channels
        offsetgradient = scalegradient = 0f0
        if layer.training
            @simd for sample in 1:observations
                value = incoming[channel, sample]
                offsetgradient += value
                scalegradient += value * normalized[channel, sample]
            end
        end
        multiplier = affine[channel, 2] * inversedeviation[channel]
        offsetmean = offsetgradient / observations
        scalemean = scalegradient / observations
        @simd for sample in 1:observations
            output[channel, sample] = multiplier * (incoming[channel, sample] -
                offsetmean - normalized[channel, sample] * scalemean)
        end
    end
    return output
end

"""Overwrite the incoming gradient with the gradient with respect to BatchNorm inputs."""
function SimpleChains.pullback_arg!(layer::SimpleChainsBatchNorm, incoming, input,
    parameters::Ptr{Float32}, memory::Ptr{UInt8}, endmemory::Ptr{UInt8})
    return _batchnormpullback!(incoming, incoming, layer, input, parameters, memory), endmemory
end

"""Write the input gradient to an explicit destination, including for a first BatchNorm layer."""
function SimpleChains.pullback_arg!(destination::Ptr{Float32}, layer::SimpleChainsBatchNorm,
    incoming, input, parameters::Ptr{Float32}, memory::Ptr{UInt8}, endmemory::Ptr{UInt8})
    output = SimpleChains.PtrArray(destination, size(input))
    return _batchnormpullback!(output, incoming, layer, input, parameters, memory), endmemory
end

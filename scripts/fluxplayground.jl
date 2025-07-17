module FluxPlayground
using DataFrames, Flux, ProgressMeter, Plots, Statistics

function binaryfluxexample()
# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),      # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2))

# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(noisy)    # 2×1000 Matrix{Float32}, or CuArray{Float32}
probs1 = softmax(out1) |> cpu    # normalise to get probabilities (and move off GPU)

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true);

opt_state = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:1_000
    for xy_cpu in loader
        # Unpack batch of data:
        x, y = xy_cpu
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

opt_state # parameters, momenta and output have all changed

out2 = model(noisy)         # first row is prob. of true, second row p(false)
probs2 = softmax(out2)      # normalise to get probabilities
m = mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!
println("mean=$m")

plotlyjs() 
# distribution plots
p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network", legend=false)

distplot = plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))
# display(distplot) # displays teh plot explicitly even inside a module context

# losses plot
plot(losses; xaxis=(:log10, "iteration"), yaxis="loss", label="per batch")
n = length(loader)
lossplot = plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)), label="epoch mean", dpi=200)
# display(lossplot) # displays teh plot explicitly even inside a module context

combinedplot = plot(distplot, lossplot, layout=(1,2))
# display(combinedplot)
return combinedplot, distplot, lossplot
end

end

cp, dp, lp = FluxPlayground.binaryfluxexample()

# lp # is not shown
# dp # is shown as return variable 
cp

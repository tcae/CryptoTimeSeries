To combine MLUtils.oversample with DataLoader in Flux.jl, you can follow these steps:

    First, import the necessary packages:

using Flux
using MLUtils

    Load your dataset and perform oversampling using MLUtils.oversample:

X, Y = load_iris()
X_bal, Y_bal = MLUtils.oversample((X, Y))

    Create a custom data loader by inheriting from DataLoader and overloading the Base.iterate method:

struct CustomDataLoader{T} <: DataLoader
    data::T
    batchsize::Int
    index::Int
end

@propagate_inbounds function Base.iterate(d::CustomDataLoader, i=0)
    if i == 0
        # Shuffle the data before iterating
        indices = shuffle(1:size(d.data[1], 1))
        d.data = (d.data[1][indices, :], d.data[2][indices])
    end

    if d.index + d.batchsize > size(d.data[1], 1)
        return nothing
    end

    batch = (d.data[1][d.index+1:d.index+d.batchsize, :], d.data[2][d.index+1:d.index+d.batchsize])
    d.index += d.batchsize

    return batch, d.index
end

    Create an instance of your custom data loader:

batchsize = 32
data_loader = CustomDataLoader((X_bal, Y_bal), batchsize, 0)

    Iterate over the data loader to get batches of data:

for (x, y) in data_loader
    # Use the batch of data for training or evaluation
    # ...
end

By following these steps, you can combine the oversampling functionality provided by MLUtils.oversample with the data loading capabilities of DataLoader in Flux.jl.

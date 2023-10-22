"""
Train and evaluate script using Flux via MLJ

https://alan-turing-institute.github.io/MLJ.jl/dev/common_mlj_workflows/#Common-MLJ-Workflows
https://1drv.ms/u/s!Alggx_XGHq4ugk8oqfO9WvjfNhgp

"""

using DataFrames, Logging  # , MLJ
using MLJ, MLJBase, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv


#region from Bing

using Flux, MLJ
# Define the network architecture
model = Chain(
Dense(12, 48, relu), # input layer with 12 signals and hidden layer with 48 nodes
Dense(48, 20, relu), # hidden layer with 20 nodes
Dense(20, 3), # output layer to classify 3 classes
softmax) # activation function for multiclass classification

# Define the loss function and the optimizer
loss(x, y) = Flux.crossentropy(model(x), y) # cross entropy loss
opt = ADAM() # adaptive moment estimation optimizer

# Load some data (replace this with your own data)
X, y = @load_iris

# Convert the data to one-hot encoding
y = Flux.onehotbatch(y, levels(y))

# Split the data into training and test sets
train, test = partition(eachindex(y), 0.8) # 80% for training, 20% for test

# Wrap the model as a MLJ machine
machine = machine(model, X, y)

# Define the resampling strategy (6-fold cross validation)
resampling = CV(nfolds=6)

# Define the evaluation metric (accuracy)
measure = accuracy

# Fit the model and evaluate it
fit!(machine, resampling=resampling, measure=measure, verbosity=1)
#endregion from Bing

#region from JuliaAI

To create a Julia deep learning network with MLJ and Flux, you can follow these steps:

    # Install the required packages:

using Pkg
Pkg.add("MLJ")
Pkg.add("MLJFlux")
Pkg.add("Flux")

    # Import the necessary modules:

using MLJ
using MLJFlux
using Flux

    # Define the structure of the network:

input_size = 12
hidden_size1 = 48
hidden_size2 = 20
output_size = 3

builder = @builder Chain(
    Dense(input_size, hidden_size1, relu),
    Dense(hidden_size1, hidden_size2, relu),
    Dense(hidden_size2, output_size)
)

    # Create the model using the defined builder:

model = MLJFlux.machine(builder)

    # Define the evaluation metric:

metric = cross_entropy

    # Define the resampling strategy:

resampling = CV(nfolds=6, shuffle=false, stratify=true)

    # Create the evaluation plan:

eval_plan = MLJ.@load Evaluator(
    model=model,
    resampling=resampling,
    measure=metric
)

    # Train and evaluate the model:

X = MLJ.table(ff)  # Assuming ff is a DataFrame
y = MLJ.coerce(ff.target, Multiclass)

mach = machine(eval_plan, X, y)
MLJ.fit!(mach)
MLJ.evaluate!(mach)

# This code creates a deep learning network with 12 input signals, a layer with 48 nodes, a layer with 20 nodes, and an output layer for 3 classes. The network is trained and evaluated using 6-fold cross-validation, with the dataset split into sequential blocks.

#endregion from JuliaAI


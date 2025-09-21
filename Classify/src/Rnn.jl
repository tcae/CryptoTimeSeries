using Flux
using Statistics
using Random

# Parameters
x = 10                      # Input features
y = 5                       # Feedback size
hidden1 = 3x
hidden2 = Int(round(2/3 * hidden1))
hidden3 = Int(round(1/2 * hidden2))
output_dim = 2              # Number of classes
seq_len = 50                # Sequence length
batch_size = 1              # For simplicity
epochs = 10

# Model components
batchnorm = BatchNorm(x + y)
dense1 = Dense(x + y, hidden1, relu)
dense2 = Dense(hidden1, hidden2, relu)
dense3 = Dense(hidden2, hidden3, relu)
rnn = LSTM(hidden3 => hidden3)
output_layer = Dense(hidden3, output_dim)

# Full model function
function recurrent_classifier(input_seq, feedback_seq)
    combined_input = hcat(input_seq, feedback_seq)
    normed = batchnorm(combined_input)
    h1 = dense1(normed)
    h2 = dense2(h1)
    h3 = dense3(h2)
    rnn_output = rnn(h3)
    return output_layer(rnn_output)
end

# Loss function
loss_fn(x, y_true, feedback) = Flux.crossentropy(recurrent_classifier(x, feedback), y_true)

# Optimizer
opt = ADAM()

# Dummy data generator
function generate_data(seq_len, x, output_dim)
    inputs = [rand(Float32, x, batch_size) for _ in 1:seq_len]
    labels = [Flux.onehot(rand(1:output_dim), 1:output_dim) for _ in 1:seq_len]
    return inputs, labels
end

sequences = [
    (input_seq1, label_seq1),
    (input_seq2, label_seq2) # ,
    # ...
]

using Random

shuffled_sequences = shuffle(sequences)

rng = MersenneTwister(42)  # Fixed seed to compare runs
shuffled_sequences = shuffle(rng, sequences)


function train_model(sequences, epochs)
    for epoch in 1:epochs
        println("Epoch $epoch")
        rnn.state = nothing  # Reset RNN state

        # Shuffle sequences for this epoch
        shuffled_sequences = shuffle(sequences)
        # Assume `sequences` is a list of tuples: (input_sequence, label_sequence)
        # Each input_sequence: Vector of Float32 arrays of shape (x, batch_size)
        # Each label_sequence: Vector of one-hot encoded labels of shape (output_dim, batch_size)

        for (input_seq, label_seq) in shuffled_sequences
            feedback_buffer = [zeros(Float32, y, batch_size) for _ in 1:y]

            for t in eachindex(input_seq)
                input_t = input_seq[t]
                label_t = label_seq[t]

                # Prepare feedback vector
                feedback_vec = reduce(hcat, feedback_buffer[end-y+1:end])
                feedback_vec = reshape(feedback_vec, y, batch_size)

                # Training step
                gs = gradient(() -> loss_fn(input_t, label_t, feedback_vec),
                              Flux.params(batchnorm, dense1, dense2, dense3, rnn, output_layer))
                Flux.Optimise.update!(opt, Flux.params(batchnorm, dense1, dense2, dense3, rnn, output_layer), gs)

                # Update feedback buffer
                pred = recurrent_classifier(input_t, feedback_vec)
                pred_class = onecold(pred, 1:output_dim)
                new_feedback = Flux.onehotbatch(pred_class, 1:output_dim) |> Array
                push!(feedback_buffer, new_feedback)
                if length(feedback_buffer) > y
                    popfirst!(feedback_buffer)
                end
            end
        end
    end
end

function load_sequences(path)
    # Load and parse each sequence file into (input_seq, label_seq)
    # Return as a list of tuples
end

sequences = load_sequences("path/to/data")
epochs=10
train_model(sequences, epochs)


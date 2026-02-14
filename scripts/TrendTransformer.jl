using Flux
using NNlib
using Statistics

# 1) Vocabularies (token IDs)
# You can keep these as Dict{Symbol,Int} and produce integer matrices shaped (seq_len, batch).

# --- Position buckets (model outputs) ---
const POS_BUCKETS = [:stronglong, :long, :flat, :choppy, :short, :strongshort]
const N_ACTIONS = length(POS_BUCKETS)

# --- Token channels (inputs) ---
# Keep 1-based IDs for Flux embeddings.
const IMPULSE = Dict(
    :up_small=>1, :up_medium=>2, :up_large=>3,
    :down_small=>4, :down_medium=>5, :down_large=>6,
    :flat=>7
)

const STRUCTURE = Dict(
    :continuation=>1, :pullback=>2, :stall=>3, :chop=>4, :coil=>5
)

const TREND_LEN = Dict(
    :trend_len_1=>1, :trend_len_2=>2, :trend_len_3_plus=>3
)

const SINCE_EVENT = Dict(
    :since_event_early=>1, :since_event_mid=>2, :since_event_late=>3
)

# Optional sparse event markers. Include a :none token for minutes with no event.
const EVENT = Dict(
    :none=>1, :E_breakout=>2, :E_trend_change=>3, :E_vol_spike=>4
)


# 2) Token-bundle embedding (one vector per minute)
# We use separate embedding tables per channel and sum them to produce a single d_model vector per minute. This is the cleanest way to keep the channels orthogonal.

struct TokenBundleEmbedding
    emb_impulse::Flux.Embedding
    emb_structure::Flux.Embedding
    emb_trendlen::Flux.Embedding
    emb_since::Flux.Embedding
    emb_event::Flux.Embedding
end

Flux.@functor TokenBundleEmbedding

"""
ids_* are Int matrices of shape (seq_len, batch)
Returns X of shape (d_model, seq_len, batch)
"""
function (E::TokenBundleEmbedding)(ids_impulse, ids_structure, ids_trendlen, ids_since, ids_event)
    d_model = size(E.emb_impulse.weight, 1)
    seq_len, batch = size(ids_impulse)

    function embed(emb, ids)
        flat = vec(ids)                             # (seq_len*batch,)
        z = emb(flat)                               # (d_model, seq_len*batch)
        reshape(z, d_model, seq_len, batch)         # (d_model, seq_len, batch)
    end

    x = embed(E.emb_impulse,  ids_impulse) .+
        embed(E.emb_structure, ids_structure) .+
        embed(E.emb_trendlen,  ids_trendlen) .+
        embed(E.emb_since,     ids_since) .+
        embed(E.emb_event,     ids_event)

    return x
end

# Constructor:

function TokenBundleEmbedding(d_model::Int)
    TokenBundleEmbedding(
        Flux.Embedding(length(IMPULSE)   => d_model),
        Flux.Embedding(length(STRUCTURE) => d_model),
        Flux.Embedding(length(TREND_LEN) => d_model),
        Flux.Embedding(length(SINCE_EVENT)=> d_model),
        Flux.Embedding(length(EVENT)     => d_model),
    )
end

# 3) ALiBi bias generation (minimal)
# Key idea
# ALiBi injects position information by adding a distance penalty to attention scores based on (query_index - key_index) with a per-head slope. [arxiv.org], [deepwiki.com]
# Flux MultiHeadAttention accepts a bias array broadcastable to (kv_len, q_len, nheads, batch) and adds it to attention scores before softmax. [fluxml.ai], [fluxml.ai]
# So we create exactly that tensor.

"""
Minimal ALiBi slopes: geometric progression per head.
DeepWiki summarizes slopes as m_h = 2^(-8h/n_heads). [4](https://deepwiki.com/ageron/handson-mlp/4.2.3-alibi:-attention-with-linear-biases)
For n_heads=8 this yields [2^-1, 2^-2, ..., 2^-8].
"""
function alibi_slopes(n_heads::Int; T=Float32)
    [T(2.0)^(-T(8) * T(h) / T(n_heads)) for h in 1:n_heads]
end

"""
Create ALiBi bias tensor for self-attention:
shape = (seq_len, seq_len, n_heads, batch)

bias[k, q, h, b] = -slope[h] * max(q-k, 0)
We rely on a causal mask to block k>q anyway.
"""
function alibi_bias(seq_len::Int, n_heads::Int, batch::Int; T=Float32)
    slopes = alibi_slopes(n_heads; T=T)                       # length n_heads
    q = reshape(T.(1:seq_len), 1, seq_len)                    # (1, L)
    k = reshape(T.(1:seq_len), seq_len, 1)                    # (L, 1)
    dist = max.(q .- k, zero(T))                               # (L, L), only past distances
    # bias per head: (L, L, H)
    B = Array{T}(undef, seq_len, seq_len, n_heads)
    for h in 1:n_heads
        @inbounds B[:, :, h] = -slopes[h] .* dist
    end
    # expand to (L, L, H, batch)
    reshape(B, seq_len, seq_len, n_heads, 1) .* ones(T, 1, 1, 1, batch)
end

# 4) Causal mask (decoder-style)
# Flux docs note the mask is applied to attention scores right before softmax, and points to NNlib.make_causal_mask. [fluxml.ai], [fluxml.ai]

"""
Create causal mask broadcastable to (kv_len, q_len, nheads, batch)
NNlib.make_causal_mask typically produces a (L, L) mask; we broadcast it.
"""
function causal_mask(seq_len::Int, n_heads::Int, batch::Int; T=Bool)
    M = NNlib.make_causal_mask(seq_len)  # (L, L) where future positions are masked
    reshape(M, seq_len, seq_len, 1, 1) .* ones(T, 1, 1, n_heads, batch)
end

# 5) Transformer block (self-attention + FFN)
# We’ll build a simple pre-norm block.

struct Block
    ln1::LayerNorm
    mha::Flux.MultiHeadAttention
    ln2::LayerNorm
    ffn::Chain
end
Flux.@functor Block

function Block(d_model::Int; n_heads::Int=8, d_ff::Int=4d_model, dropout_p::Float32=0.1f0)
    mha = Flux.MultiHeadAttention(d_model; nheads=n_heads, dropout_prob=dropout_p)
    ffn = Chain(
        Dense(d_model => d_ff, gelu),
        Dropout(dropout_p),
        Dense(d_ff => d_model),
    )
    Block(LayerNorm(d_model), mha, LayerNorm(d_model), ffn)
end

"""
x: (d_model, seq_len, batch)
bias: (seq_len, seq_len, n_heads, batch)
mask: (seq_len, seq_len, n_heads, batch)
"""
function (b::Block)(x; bias=nothing, mask=nothing)
    h = b.ln1(x)
    # mha returns (y, attn_weights)
    y, _ = b.mha(h, h, h, bias; mask=mask)  # bias added before softmax; mask applied before softmax [1](https://fluxml.ai/FastAI.jl/dev/Flux@0.13.17/src/layers/attention.jl.html)[2](https://fluxml.ai/FastAI.jl/dev/Flux@0.13.17/ref/Flux.MultiHeadAttention.html)
    x = x .+ y
    x = x .+ b.ffn(b.ln2(x))
    return x
end

# 6) Full model: token embeddings → N blocks → decision head
# Your action space is 6 buckets:
# stronglong, long, flat, choppy, short, strongshort
# We’ll classify based on the latest minute representation (last position in the window).

struct TradingTransformer
    embed::TokenBundleEmbedding
    blocks::Vector{Block}
    head::Dense
    n_heads::Int
end
Flux.@functor TradingTransformer

function TradingTransformer(d_model::Int=128; n_layers::Int=4, n_heads::Int=8, d_ff::Int=512, dropout_p::Float32=0.1f0)
    embed = TokenBundleEmbedding(d_model)
    blocks = [Block(d_model; n_heads=n_heads, d_ff=d_ff, dropout_p=dropout_p) for _ in 1:n_layers]
    head = Dense(d_model => N_ACTIONS)  # logits for 6 buckets
    TradingTransformer(embed, blocks, head, n_heads)
end

"""
Forward pass:
ids_* are (seq_len, batch)
Returns logits (N_ACTIONS, batch)
"""
function (m::TradingTransformer)(ids_impulse, ids_structure, ids_trendlen, ids_since, ids_event)
    # Embed token bundle -> (d_model, seq_len, batch)
    x = m.embed(ids_impulse, ids_structure, ids_trendlen, ids_since, ids_event)

    seq_len, batch = size(ids_impulse)

    # ALiBi bias and causal mask
    B = alibi_bias(seq_len, m.n_heads, batch)       # (L, L, H, B)
    M = causal_mask(seq_len, m.n_heads, batch)      # (L, L, H, B)

    # Transformer blocks
    for blk in m.blocks
        x = blk(x; bias=B, mask=M)
    end

    # Decision head on most recent timestep
    x_last = x[:, end, :]                    # (d_model, batch)
    logits = m.head(x_last)                  # (N_ACTIONS, batch)
    return logits
end

# 7) Loss function + example training step (skeleton)
# Your labels should be integers 1..6 corresponding to POS_BUCKETS.

# Cross entropy for logits and class indices
lossfn(logits, y) = Flux.logitcrossentropy(logits, Flux.onehotbatch(y, 1:N_ACTIONS))

# Example one training step
function train_step!(model, opt, batch)
    (ids_impulse, ids_structure, ids_trendlen, ids_since, ids_event, y) = batch

    gs = gradient(Flux.params(model)) do
        logits = model(ids_impulse, ids_structure, ids_trendlen, ids_since, ids_event)
        lossfn(logits, y)
    end

    Flux.Optimise.update!(opt, Flux.params(model), gs)
end

# Why this matches our design precisely

# No positional embeddings are added to input vectors. Position is injected via a bias on attention logits, consistent with ALiBi’s definition (“bias query-key attention scores with a distance-proportional penalty”). [arxiv.org], [deepwiki.com]
# Flux’s MultiHeadAttention explicitly supports a bias tensor that is added to attention scores before softmax, and a mask that is applied before softmax. 
# That’s exactly the hook ALiBi needs. [fluxml.ai], [fluxml.ai]


# Practical notes (based on your earlier choices)


# since_event_* and trend_len_* stay as semantic tokens
# ALiBi does sequence order, your tokens do market time semantics.


# Your “choppy” bucket is an output action, not an input token
# It’s fine to interpret it as “prefer mean-reversion exit behavior”; keep execution logic outside the model.
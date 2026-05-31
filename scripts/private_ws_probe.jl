"""
private_ws_probe.jl

Low-level private websocket probe for KrakenSpot and KrakenFutures.
This script bypasses the trading loop and prints raw receive diagnostics,
so handshake/read failures can be isolated from strategy/order logic.

Usage examples:
  julia --project=scripts scripts/private_ws_probe.jl
  julia --project=scripts scripts/private_ws_probe.jl xch=KrakenSpot
  julia --project=scripts scripts/private_ws_probe.jl xch=KrakenFutures channel=orders
  julia --project=scripts scripts/private_ws_probe.jl stack=websockets

Arguments:
  xch=KrakenSpot|KrakenFutures|both
  channel=orders|balances|both
  stack=http|websockets|both
  messages=<positive integer>
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, HTTP, JSON3
using KrakenSpot, KrakenFutures

"Return value for key from key=value args, or default when missing."
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing, String}=nothing)
    prefix = String(key) * "="
    for arg in args
        startswith(arg, prefix) || continue
        return strip(arg[(length(prefix)+1):end])
    end
    return default
end

"Timestamped log line for probe output."
function _logline(msg::AbstractString)
    println("[", Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z] ", msg)
    return nothing
end

"Parse positive integer argument with clear assertion text."
function _parsepositiveint(raw::AbstractString, key::AbstractString)::Int
    value = parse(Int, raw)
    @assert value > 0 "$(key) must be > 0, got $(value)"
    return value
end

"Return a string preview for a potentially non-string value."
function _stringpreview(value; limit::Int=240)
    if value === nothing
        return nothing
    elseif value isa AbstractString
        return first(value, min(lastindex(value), limit))
    elseif value isa AbstractVector{UInt8}
        text = try
            String(Vector{UInt8}(value))
        catch
            repr(value)
        end
        return first(text, min(lastindex(text), limit))
    end
    representation = repr(value)
    return first(representation, min(lastindex(representation), limit))
end

"Safely read a property when the underlying exception type exposes it."
function _maybeproperty(value, name::Symbol)
    try
        return hasproperty(value, name) ? getproperty(value, name) : nothing
    catch
        return nothing
    end
end

"Build a compact diagnostic payload for websocket and HTTP failures."
function _failureinfo(err)
    info = Dict{String, Any}(
        "exception_type" => string(typeof(err)),
        "exception_message" => sprint(showerror, err),
    )

    for name in (:status, :code, :closecode, :close_code, :reason, :close_reason, :closereason, :message)
        value = _maybeproperty(err, name)
        if !isnothing(value)
            info[string(name)] = _stringpreview(value)
        end
    end

    response = _maybeproperty(err, :response)
    if !isnothing(response)
        responseinfo = Dict{String, Any}("type" => string(typeof(response)))
        for name in (:status, :reason, :headers, :body)
            value = _maybeproperty(response, name)
            if !isnothing(value)
                responseinfo[string(name)] = name == :body ? _previewpayload(value) : _stringpreview(value)
            end
        end
        info["response"] = responseinfo
    end

    return info
end

"Convert websocket receive payload to printable preview and metadata."
function _previewpayload(msgraw)
    if msgraw isa String
        text = msgraw
        return Dict(
            "type" => "String",
            "len" => lastindex(text),
            "starts_http101" => startswith(text, "HTTP/1.1 101"),
            "preview" => first(text, min(lastindex(text), 240)),
        )
    elseif msgraw isa AbstractVector{UInt8}
        bytes = Vector{UInt8}(msgraw)
        text = try
            String(bytes)
        catch
            "<non-utf8-binary>"
        end
        return Dict(
            "type" => "Vector{UInt8}",
            "len" => length(bytes),
            "starts_http101" => startswith(text, "HTTP/1.1 101"),
            "preview" => first(text, min(lastindex(text), 240)),
            "bytes_head" => join(string.(bytes[1:min(end, 24)]), ","),
        )
    end
    return Dict("type" => string(typeof(msgraw)), "repr" => repr(msgraw))
end

"Return the websocket module imported by the selected exchange module."
function _wsmod(xch::Symbol)
    if xch == :krakenspot
        return KrakenSpot.WebSockets
    elseif xch == :krakenfutures
        return KrakenFutures.WebSockets
    end
    error("unknown exchange=$(xch)")
end

"Open websocket URL with selected stack and invoke callback with open socket."
function _wsopen(stack::Symbol, wsmod, url::String, handler::Function)
    stackname = stack == :http ? "HTTP.WebSockets" : stack == :websockets ? "WebSockets" : string(stack)
    _logline("ws open stack=$(stackname) url=$(url)")
    if stack == :http
        try
            return HTTP.WebSockets.open(url) do ws
                handler(ws)
            end
        catch err
            _logline("ws open failed stack=$(stackname) url=$(url) info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
    elseif stack == :websockets
        try
            return wsmod.open(url) do ws
                handler(ws)
            end
        catch err
            _logline("ws open failed stack=$(stackname) url=$(url) info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
    end
    error("unknown stack=$(stack)")
end

"Send payload over selected websocket stack."
function _wssend(stack::Symbol, wsmod, ws, payload::String)
    if stack == :http
        HTTP.WebSockets.send(ws, payload)
    elseif stack == :websockets
        wsmod.send(ws, payload)
    else
        error("unknown stack=$(stack)")
    end
    return nothing
end

"Receive one payload over selected websocket stack."
function _wsreceive(stack::Symbol, wsmod, ws)
    if stack == :http
        return HTTP.WebSockets.receive(ws)
    elseif stack == :websockets
        return wsmod.receive(ws)
    end
    error("unknown stack=$(stack)")
end

"Read and print a fixed number of incoming frames for the current subscription."
function _readframes!(stack::Symbol, wsmod, ws, nmessages::Int)
    for i in 1:nmessages
        try
            msgraw = _wsreceive(stack, wsmod, ws)
            info = _previewpayload(msgraw)
            _logline("recv[$(i)] $(JSON3.write(info))")
        catch err
            _logline("recv[$(i)] failed info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
    end
    return nothing
end

"Probe KrakenSpot private websocket for one channel using selected websocket stack."
function _probe_krakenspot_channel(stack::Symbol, channel::String, nmessages::Int)
    wsmod = _wsmod(:krakenspot)
    bc = KrakenSpot.KrakenSpotCache()
    token = KrakenSpot._wsauthtoken(bc)
    @assert !isnothing(token) "KrakenSpot websocket token request returned nothing"

    subscribe = Dict("method" => "subscribe", "params" => Dict("channel" => channel, "token" => token))
    _logline("KrakenSpot stack=$(stack) channel=$(channel) opening $(KrakenSpot.KRAKEN_WS_PRIVATE)")

    _wsopen(stack, wsmod, KrakenSpot.KRAKEN_WS_PRIVATE, ws -> begin
        _wssend(stack, wsmod, ws, JSON3.write(subscribe))
        _logline("KrakenSpot subscribe sent for channel=$(channel)")
        _readframes!(stack, wsmod, ws, nmessages)
    end)
    return nothing
end

"Probe KrakenFutures private websocket for one feed using selected websocket stack."
function _probe_krakenfutures_feed(stack::Symbol, feed::String, nmessages::Int)
    wsmod = _wsmod(:krakenfutures)
    bc = KrakenFutures.KrakenFuturesCache()
    @assert !isempty(strip(String(bc.publickey))) "KrakenFutures public key is empty"
    @assert !isempty(strip(String(bc.secretkey))) "KrakenFutures secret key is empty"

    _logline("KrakenFutures stack=$(stack) feed=$(feed) opening $(KrakenFutures.KRAKEN_FUTURES_WS_PRIVATE)")

    _wsopen(stack, wsmod, KrakenFutures.KRAKEN_FUTURES_WS_PRIVATE, ws -> begin
        challreq = Dict("event" => "challenge", "api_key" => bc.publickey)
        _wssend(stack, wsmod, ws, JSON3.write(challreq))
        _logline("KrakenFutures challenge request sent")

        challraw = try
            _wsreceive(stack, wsmod, ws)
        catch err
            _logline("KrakenFutures challenge receive failed info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
        challinfo = _previewpayload(challraw)
        _logline("KrakenFutures challenge recv $(JSON3.write(challinfo))")

        challtxt = challraw isa String ? challraw : String(Vector{UInt8}(challraw))
        challmsg = JSON3.read(challtxt, Dict)
        challenge = haskey(challmsg, "message") ? String(challmsg["message"]) : nothing
        @assert !isnothing(challenge) "KrakenFutures challenge missing message field: $(challtxt)"

        subscribe = Dict(
            "event" => "subscribe",
            "feed" => feed,
            "api_key" => bc.publickey,
            "original_challenge" => challenge,
            "signed_challenge" => KrakenFutures._wssignedchallenge(bc.secretkey, challenge),
        )
        _wssend(stack, wsmod, ws, JSON3.write(subscribe))
        _logline("KrakenFutures subscribe sent for feed=$(feed)")
        _readframes!(stack, wsmod, ws, nmessages)
    end)
    return nothing
end

"Run one probe target and print full error diagnostics on failure."
function _runprobe(label::String, f::Function)
    _logline("START $(label)")
    try
        f()
        _logline("PASS  $(label)")
    catch err
        _logline("FAIL  $(label) err=$(typeof(err)) msg=$(sprint(showerror, err))")
        Base.showerror(stdout, err, catch_backtrace())
        println()
    end
    return nothing
end

"Main script entrypoint."
function main(args::Vector{String})
    xch_raw = lowercase(strip(_argvalue(args, "xch", "both")))
    channel_raw = lowercase(strip(_argvalue(args, "channel", "both")))
    stack_raw = lowercase(strip(_argvalue(args, "stack", "both")))
    messages = _parsepositiveint(_argvalue(args, "messages", "3"), "messages")

    exchanges = xch_raw == "both" ? ["krakenspot", "krakenfutures"] : [xch_raw]
    channels = channel_raw == "both" ? ["orders", "balances"] : [channel_raw]
    stacks = stack_raw == "both" ? [:http, :websockets] : [Symbol(stack_raw)]

    @assert all(x -> x in ["krakenspot", "krakenfutures"], exchanges) "xch must be KrakenSpot|KrakenFutures|both, got $(xch_raw)"
    @assert all(c -> c in ["orders", "balances"], channels) "channel must be orders|balances|both, got $(channel_raw)"
    @assert all(s -> s in [:http, :websockets], stacks) "stack must be http|websockets|both, got $(stack_raw)"

    _logline("private_ws_probe start xch=$(xch_raw) channel=$(channel_raw) stack=$(stack_raw) messages=$(messages)")

    for stack in stacks
        for xch in exchanges
            for channel in channels
                if xch == "krakenspot"
                    ws_channel = channel == "orders" ? "executions" : "balances"
                    label = "KrakenSpot stack=$(stack) channel=$(ws_channel)"
                    _runprobe(label, () -> begin
                        _probe_krakenspot_channel(stack, ws_channel, messages)
                    end)
                else
                    ws_feed = channel == "orders" ? "open_orders" : "balances"
                    label = "KrakenFutures stack=$(stack) feed=$(ws_feed)"
                    _runprobe(label, () -> begin
                        _probe_krakenfutures_feed(stack, ws_feed, messages)
                    end)
                end
            end
        end
    end

    _logline("private_ws_probe finished")
    return nothing
end

main(ARGS)

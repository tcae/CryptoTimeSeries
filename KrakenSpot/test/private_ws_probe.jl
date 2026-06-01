"""
private_ws_probe.jl

Low-level private websocket probe for KrakenSpot and KrakenFutures.
This script bypasses the trading loop and prints raw receive diagnostics,
so handshake/read failures can be isolated from strategy/order logic.

Usage examples:
    julia KrakenSpot/test/private_ws_probe.jl xch=KrakenSpot
    julia KrakenSpot/test/private_ws_probe.jl xch=KrakenFutures channel=orders
    julia KrakenSpot/test/private_ws_probe.jl xch=KrakenFutures stack=websockets
    julia KrakenSpot/test/private_ws_probe.jl xch=both

Arguments:
  xch=KrakenSpot|KrakenFutures|both
  channel=orders|balances|both
  stack=http|websockets|both
  messages=<positive integer>
"""

import Pkg

"Return value for key from key=value args, or default when missing."
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing, String}=nothing)
    prefix = String(key) * "="
    for arg in args
        startswith(arg, prefix) || continue
        return strip(arg[(length(prefix)+1):end])
    end
    return default
end

"Choose bootstrap project to avoid legacy scripts HTTP/WebSockets pin when possible."
function _bootstrapproject(args::Vector{String})::String
    xch_raw = lowercase(strip(_argvalue(args, "xch", "both")))
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    if xch_raw == "krakenspot"
        return normpath(joinpath(@__DIR__, ".."))
    elseif xch_raw == "krakenfutures"
        return normpath(joinpath(repo_root, "KrakenFutures"))
    end
    return normpath(joinpath(repo_root, "scripts"))
end

const _BOOTSTRAP_PROJECT = _bootstrapproject(ARGS)
Pkg.activate(_BOOTSTRAP_PROJECT, io=devnull)

using Dates, HTTP, JSON3

const _BOOTSTRAP_XCH = lowercase(strip(_argvalue(ARGS, "xch", "both")))
if _BOOTSTRAP_XCH in ("krakenspot", "both")
    @eval using KrakenSpot
end
if _BOOTSTRAP_XCH in ("krakenfutures", "both")
    @eval using KrakenFutures
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

"Throw clear error when websocket payload reports alert/error semantics."
function _throwonwserror(msgraw)
    msgtxt = nothing
    if msgraw isa String
        msgtxt = msgraw
    elseif msgraw isa AbstractVector{UInt8}
        msgtxt = try
            String(Vector{UInt8}(msgraw))
        catch
            nothing
        end
    end
    isnothing(msgtxt) && return nothing

    parsed = try
        JSON3.read(msgtxt, Dict)
    catch
        return nothing
    end

    event = haskey(parsed, "event") ? lowercase(String(parsed["event"])) : ""
    if event in ("alert", "error")
        throw(ErrorException("websocket $(event): $(msgtxt)"))
    end
    if haskey(parsed, "success") && parsed["success"] === false
        throw(ErrorException("websocket subscribe failure: $(msgtxt)"))
    end
    return nothing
end

"Read and print a fixed number of incoming frames for the current subscription."
function _readframes!(stack::Symbol, wsmod, ws, nmessages::Int)
    for i in 1:nmessages
        try
            msgraw = _wsreceive(stack, wsmod, ws)
            info = _previewpayload(msgraw)
            _logline("recv[$(i)] $(JSON3.write(info))")
            _throwonwserror(msgraw)
        catch err
            _logline("recv[$(i)] failed info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
    end
    return nothing
end

"Build KrakenSpot cache with exchange-scoped EnvConfig credentials when available."
function _krakenspotcache()
    auth = try
        KrakenSpot.EnvConfig.Authentication(nothing; exchange="KrakenSpot")
    catch
        nothing
    end
    bc = isnothing(auth) ? KrakenSpot.KrakenSpotCache() : KrakenSpot.KrakenSpotCache(publickey=String(auth.key), secretkey=String(auth.secret))
    @assert !isempty(strip(String(bc.publickey))) "KrakenSpot public key is empty"
    @assert !isempty(strip(String(bc.secretkey))) "KrakenSpot secret key is empty"
    return bc
end

"Receive KrakenFutures websocket challenge frame, skipping initial non-challenge frames."
function _krakenfutureschallenge!(stack::Symbol, wsmod, ws)::String
    for i in 1:10
        challraw = try
            _wsreceive(stack, wsmod, ws)
        catch err
            _logline("KrakenFutures challenge receive failed info=$(JSON3.write(_failureinfo(err)))")
            rethrow()
        end
        challinfo = _previewpayload(challraw)
        _logline("KrakenFutures challenge recv[$(i)] $(JSON3.write(challinfo))")

        challtxt = challraw isa String ? challraw : String(Vector{UInt8}(challraw))
        challmsg = JSON3.read(challtxt, Dict)
        if haskey(challmsg, "message")
            return String(challmsg["message"])
        end
    end
    throw(AssertionError("KrakenFutures challenge message not received after 10 frames"))
end

"Run one exchange/channel probe based on normalized routing names."
function _probetarget(stack::Symbol, xch::String, channel::String, nmessages::Int)
    if xch == "krakenspot"
        ws_channel = channel == "orders" ? "executions" : "balances"
        _runprobe("KrakenSpot stack=$(stack) channel=$(ws_channel)", () -> _probe_krakenspot_channel(stack, ws_channel, nmessages))
    else
        ws_feed = channel == "orders" ? "open_orders" : "balances"
        _runprobe("KrakenFutures stack=$(stack) feed=$(ws_feed)", () -> _probe_krakenfutures_feed(stack, ws_feed, nmessages))
    end
    return nothing
end

"Probe a private websocket target with a one-step connect, send, and read flow."
function _probeprivate(stack::Symbol, wsmod, url::String, prepare::Function, receivefirst::Function, nmessages::Int)
    _wsopen(stack, wsmod, url, ws -> begin
        prepare(ws)
        _readframes!(stack, wsmod, ws, nmessages)
    end)
    return nothing
end

"Probe KrakenSpot private websocket for one channel using selected websocket stack."
function _probe_krakenspot_channel(stack::Symbol, channel::String, nmessages::Int)
    wsmod = _wsmod(:krakenspot)
    bc = _krakenspotcache()
    token = KrakenSpot._wsauthtoken(bc)
    @assert !isnothing(token) "KrakenSpot websocket token request returned nothing"

    subscribe = Dict("method" => "subscribe", "params" => Dict("channel" => channel, "token" => token))
    _logline("KrakenSpot stack=$(stack) channel=$(channel) opening $(KrakenSpot.KRAKEN_WS_PRIVATE)")

    _probeprivate(stack, wsmod, KrakenSpot.KRAKEN_WS_PRIVATE, ws -> begin
        _wssend(stack, wsmod, ws, JSON3.write(subscribe))
        _logline("KrakenSpot subscribe sent for channel=$(channel)")
    end, ws -> nothing, nmessages)
    return nothing
end

"Probe KrakenFutures private websocket for one feed using selected websocket stack."
function _probe_krakenfutures_feed(stack::Symbol, feed::String, nmessages::Int)
    wsmod = _wsmod(:krakenfutures)
    bc = KrakenFutures.KrakenFuturesCache()
    @assert !isempty(strip(String(bc.publickey))) "KrakenFutures public key is empty"
    @assert !isempty(strip(String(bc.secretkey))) "KrakenFutures secret key is empty"

    _logline("KrakenFutures stack=$(stack) feed=$(feed) opening $(KrakenFutures.KRAKEN_FUTURES_WS_PRIVATE)")

    _probeprivate(stack, wsmod, KrakenFutures.KRAKEN_FUTURES_WS_PRIVATE, ws -> begin
        challreq = Dict("event" => "challenge", "api_key" => bc.publickey)
        _wssend(stack, wsmod, ws, JSON3.write(challreq))
        _logline("KrakenFutures challenge request sent")
        challenge = _krakenfutureschallenge!(stack, wsmod, ws)
        subscribe = Dict("event" => "subscribe", "feed" => feed, "api_key" => bc.publickey, "original_challenge" => challenge, "signed_challenge" => KrakenFutures._wssignedchallenge(bc.secretkey, challenge))
        _wssend(stack, wsmod, ws, JSON3.write(subscribe))
        _logline("KrakenFutures subscribe sent for feed=$(feed)")
    end, ws -> nothing, nmessages)
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
                _probetarget(stack, xch, channel, messages)
            end
        end
    end

    _logline("private_ws_probe finished")
    return nothing
end

main(ARGS)

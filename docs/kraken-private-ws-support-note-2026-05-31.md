# Kraken Private WebSocket Support Note (2026-05-31)

## Summary

We reproduced the private websocket issue on Kraken Futures using an isolated probe outside the trading loop.

The failure occurs during websocket open/handshake processing, before any websocket close frame is observed. As a result, we do not have a private socket close code or close reason to report yet.

## Current verified behavior

- The issue is reproducible on the private websocket path.
- The probe reaches an `HTTP/1.1 101 Switching Protocols` upgrade response.
- The client then fails in the `HTTP` / `WebSockets.jl` open/read path with a `MethodError(convert, ...)`.
- No websocket close code or close reason is emitted before the failure.
- Public websocket behavior was not the focus of this probe; the reproduced failure is on the private path.

## Client and library path

- Probe script: `scripts/private_ws_probe.jl`
- Futures probe path: `WebSockets.open(...)` through the `WebSockets` module imported by `KrakenFutures`
- Underlying stack in the failure trace: `HTTP.ConnectionPool` -> `HTTP.Streams` -> `WebSockets`

## Reproduction command

```bash
CTS_WS_ORDERS_ENABLED=true CTS_WS_BALANCES_ENABLED=true \
julia --project=scripts scripts/private_ws_probe.jl \
  xch=KrakenFutures channel=both stack=websockets messages=1
```

## Captured probe metadata

- Start timestamp: `2026-05-31T16:56:05Z`
- Failure timestamp: `2026-05-31T16:56:09Z`
- Public IP: `86.83.168.195`
- Region: `Amsterdam, North Holland, NL`
- Julia: `1.12.6`
- HTTP.jl: `0.9.17`
- WebSockets.jl: `1.5.9`
- Endpoint used: `wss://futures.kraken.com/ws/v1`
- REST base associated with the probe: `https://futures.kraken.com/derivatives/api/v3`

## Observed failure

```text
MethodError: Cannot `convert` an object of type
  SubArray{UInt8,1,Memory{UInt8},Tuple{UnitRange{Int64}},true}
to an object of type
  SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{Int64}},true}
```

The stack trace shows the failure originating in `HTTP.IOExtras.readuntil` while `WebSockets.open` is processing the upgraded connection.

## What to share with support

- The private socket uses `WebSockets.jl` in the probe path.
- The failure is pre-close, so there is no close code/reason available from this reproduction.
- The upgrade response is received, but the client fails during stream/header processing immediately afterward.
- If they want to correlate further, the exact reproduction command above is the cleanest way to reproduce the issue.

## Kraken guidance to include

- Kraken Futures documented websocket endpoint: `wss://futures.kraken.com/ws/v1`
- Kraken Futures REST base: `https://futures.kraken.com/derivatives/api/v3`
- If a probe is hitting `ws-auth.kraken.com`, that is the Spot private websocket URL, not the Futures endpoint.
- Kraken does not have published guidance for the Julia `MethodError(convert, SubArray...)` that appears after HTTP 101.
- For escalation, include:
  - timestamp(s)
  - endpoint used
  - full HTTP upgrade request and response headers
  - public IP and region
  - any Cloudflare identifiers if present
- If a Cloudflare error page appears, also capture:
  - `RayID`
  - `colo=` from the trace page

## Cloudflare identifiers observed in live logs

- Example CF-RAY values seen in HTTP 101 header dumps:
  - `a04795f3c81a0adc-AMS`
  - `a047960b9c416650-AMS`
  - `a0479624fa080e94-AMS`
  - `a047966f1c8efe9e-AMS`
- Observed colo suffix from these headers: `AMS`
- Observation window from live loop snippet: `2026-05-31T17:09:25Z` to `2026-05-31T17:09:50Z`
- Note: these values come from response headers in the runtime logs, not from a Cloudflare trace page.

## Notes

- The probe was intentionally isolated from strategy and trade execution logic.
- The current result is useful for support because it separates transport-level failure from order logic and avoids the noise of the live trading loop.
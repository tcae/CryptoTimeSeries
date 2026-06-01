# Private WebSocket Probe Support Note (2026-06-01)

## Scope

This note summarizes current private websocket probe behavior for KrakenSpot and KrakenFutures using the same probe entrypoint:

- `scripts/private_ws_probe.jl` (wrapper)
- canonical implementation: `KrakenSpot/test/private_ws_probe.jl`

All results below are from fresh runs on 2026-06-01.

## Reproduction Commands

```bash
# KrakenSpot private WS (executions + balances), both stack paths
julia scripts/private_ws_probe.jl xch=KrakenSpot channel=both stack=both messages=3

# KrakenFutures private WS (open_orders + balances), both stack paths
julia scripts/private_ws_probe.jl xch=KrakenFutures channel=both stack=both messages=3
```

## Environment Notes

- KrakenSpot and KrakenFutures probe paths both use `HTTP.WebSockets` API through module aliasing.
- KrakenSpot probe now explicitly sources credentials from `EnvConfig.Authentication(...; exchange="KrakenSpot")` before building cache.
- KrakenFutures package HTTP version was aligned to the KrakenSpot-confirmed version:
  - KrakenSpot: `HTTP v1.11.0`
  - KrakenFutures: `HTTP v1.11.0`

## What Works

### KrakenSpot private websocket

Status: **PASS** in both stack modes (`http`, `websockets`) for both channels (`executions`, `balances`).

Observed sequence (representative):

- WS connect/open succeeds to `wss://ws-auth.kraken.com/v2`
- status update frame received
- subscribe ack frame received (`"success": true`)
- snapshot frame received for requested channel

Examples from probe output:

- `PASS  KrakenSpot stack=http channel=executions`
- `PASS  KrakenSpot stack=http channel=balances`
- `PASS  KrakenSpot stack=websockets channel=executions`
- `PASS  KrakenSpot stack=websockets channel=balances`

## What Still Fails

### KrakenFutures private websocket authenticated feeds

Status: **FAIL** in both stack modes (`http`, `websockets`) for both feeds (`open_orders`, `balances`).

Observed sequence (representative):

- WS connect/open succeeds to `wss://futures.kraken.com/ws/v1`
- challenge request is accepted
- challenge response frame with `"event":"challenge"` and `"message": ...` is received
- subscribe request is sent
- first feed frame returns:
  - `{"event":"alert","message":"Failed to subscribe to authenticated feed"}`

Because probe now treats websocket `alert/error` responses as failures, each of these runs is marked FAIL.

Examples from probe output:

- `FAIL  KrakenFutures stack=http feed=open_orders`
- `FAIL  KrakenFutures stack=http feed=balances`
- `FAIL  KrakenFutures stack=websockets feed=open_orders`
- `FAIL  KrakenFutures stack=websockets feed=balances`

## Interpretation by Context

- **Transport context**: websocket transport/open/challenge path is functional on KrakenFutures.
- **Authenticated subscription context**: feed subscription for private futures channels is still rejected by exchange with explicit alert.
- **Version-control context**: aligning KrakenFutures to `HTTP v1.11.0` (same as KrakenSpot) did not change this failure mode.

## Suggested Support Focus

For KrakenFutures private WS support escalation, include:

- timestamped probe runs
- endpoint: `wss://futures.kraken.com/ws/v1`
- challenge request/response success evidence
- subscribe payload shape (`open_orders`, `balances`, `api_key`, `original_challenge`, `signed_challenge`)
- alert response payload: `Failed to subscribe to authenticated feed`
- note that the same key can authenticate private REST endpoints, but private WS feed subscribe is rejected

## Local Artifact Paths

- `/tmp/krakenspot_probe_20260601.log`
- `/tmp/krakenfutures_probe_20260601.log`

# KrakenFutures Support Package (2026-05-30)

## Email Version

Subject: Update: Kraken Futures POST authentication now succeeds on safe probe

Body:

Hello Kraken Support,

I am sharing an update after following your guidance (omit `Nonce` or use continuously increasing milliseconds nonce).

Using your recommendation, authenticated private POST requests now succeed at the auth layer on a safe probe endpoint and return application-level validation (`Invalid UUID`) instead of nonce/authentication errors.

Safe repro endpoint:

`POST https://futures.kraken.com/derivatives/api/v3/orders/status`

Probe body:

`orderIds=non-existent-order-id`

Observed response:

`{"status":"BAD_REQUEST","result":"error","errors":[{"code":11,"message":"Invalid UUID string: non-existent-order-id"}],...}`

I also verified that private `GET /accounts` succeeds with the same credentials.

Support confirmed:

- `Nonce` is optional for private endpoints.
- If `Nonce` is provided, use continuously increasing UNIX time in milliseconds.
- The signed `endpointPath` for `Authent` is `/api/v3/...` while requests are sent to `/derivatives/api/v3/...`.

Thank you.

## Summary

We can authenticate successfully against Kraken Futures private GET endpoints with the configured KrakenFutures API key, and authenticated private POST requests now also authenticate successfully on a safe probe endpoint.

The external shell-only `curl` probe now reaches endpoint-level validation and returns `Invalid UUID string` on `/derivatives/api/v3/orders/status`, which indicates auth success.

## Current verified behavior

- Exchange: Kraken Futures
- Base URL: `https://futures.kraken.com`
- Safe POST probe endpoint: `/derivatives/api/v3/orders/status`
- Probe body: `orderIds=non-existent-order-id`
- GET `/accounts` succeeds with the same credentials in the client.
- POST `/orders/status` now authenticates and returns endpoint validation: `Invalid UUID string: non-existent-order-id`.
- Verified with both:
  - shell-only `curl` + local signature generation
  - direct Julia `Downloads.request` call
  - `HttpPrivateRequest` after aligning private POST transport with the validated direct path

## Shell-only reproduction

This version is redacted and uses environment variables only.

```bash
export KRAKEN_FUTURES_API_KEY='YOUR_API_KEY'
export KRAKEN_FUTURES_API_SECRET='YOUR_BASE64_SECRET'

export URL='https://futures.kraken.com/derivatives/api/v3/orders/status'
export SIG_ENDPOINT='/api/v3/orders/status'
export POST_DATA='orderIds=non-existent-order-id'
export NONCE="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"

export AUTHENT="$(python3 - <<'PY'
import base64, hashlib, hmac, os
secret = base64.b64decode(os.environ['KRAKEN_FUTURES_API_SECRET'])
message = hashlib.sha256((os.environ['POST_DATA'] + os.environ['NONCE'] + os.environ['SIG_ENDPOINT']).encode()).digest()
print(base64.b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode())
PY
)"

curl -sS -X POST "$URL" \
  -H "APIKey: $KRAKEN_FUTURES_API_KEY" \
  -H "Nonce: $NONCE" \
  -H "Authent: $AUTHENT" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data "$POST_DATA"
```

## Observed response from local validation

```json
{"status":"BAD_REQUEST","result":"error","errors":[{"code":11,"message":"Invalid UUID string: non-existent-order-id"}],"serverTime":"2026-05-30T13:01:35.383Z"}
```

## Why this probe is useful

- It is a POST, so it exercises authenticated private POST behavior.
- It is safer than `/sendorder` because it queries order status for a non-existent ID instead of placing an order.
- It reproduces the same auth outcome outside the full trading loop and without relying on our higher-level trade flow.

## Notes on example ambiguity

We checked Kraken's public Go client `krakenfx/api-go` while building this probe.

- The request URL path is `/derivatives/api/v3/...`.
- The Go client signs `request.URL.Path` after trimming the `/derivatives` prefix, so the effective signed endpoint path is `/api/v3/...`.
- Some documentation examples appear to describe signing with `/derivatives/api/v3/...` directly.

Our reproduced probe uses `SIG_ENDPOINT=/api/v3/orders/status`, which matches Kraken's public Go implementation.

## Support-confirmed signing and nonce rules

Kraken support confirmed the following:

- `Nonce` is optional for private endpoints.
- If provided, `Nonce` should be continuously increasing UNIX milliseconds.
- `Authent` must be signed with `endpointPath` in `/api/v3/...` form.
- Request URLs remain under `/derivatives/api/v3/...`.

## Additional internal findings

- We fixed a local credential-selection bug so the KrakenFutures adapter now uses the exchange-scoped KrakenFutures auth entry rather than a global default auth entry.
- We fixed form encoding for repeated string fields like `orderIds` so the probe body is now encoded as repeated form parameters rather than a single bracketed string.
- We added startup protection against obviously poisoned persisted nonce floors that were far ahead of wall clock.
- We aligned nonce mode with Kraken guidance (ms by default, with optional ns mode only by explicit env setting).
- We aligned private POST transport in `HttpPrivateRequest` with the validated direct probe path.

Those fixes changed behavior from generic `authenticationError` and nonce errors to stable authenticated endpoint-level validation on the safe POST probe.
"""
Compatibility wrapper for the websocket probe.

Canonical probe location:
  KrakenSpot/test/private_ws_probe.jl
"""

include(normpath(joinpath(@__DIR__, "..", "KrakenSpot", "test", "private_ws_probe.jl")))

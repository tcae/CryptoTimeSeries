using Test
using Dates
using DataFrames
using CategoricalArrays
using Targets
using TSM
using TradingStrategy

# Pins the selective-carry contract of TradingStrategy._rowtakeover!.
#
# _rowtakeover! copies 29 order/position fields from row ix-1 to row ix and leaves every
# other Trades column untouched. It is NOT a whole-row copy: opentime/high/low/close/score/
# label are prepopulated per row by the replay input, and carrying them would overwrite that
# input with the previous row's values. Any port of the replay row loop to hoisted column
# handles or lastrow/newrow structs must reproduce exactly this partition, so it is asserted
# here against an independent reference implementation rather than left to review.

"""Fields `_rowtakeover!` carries from row `ix-1` to row `ix`."""
const ROWTAKEOVER_CARRIED = Symbol[
    :lo_limit, :lc_limit, :so_limit, :sc_limit,
    :lcsl_limit, :scsl_limit,
    :lo_amount, :lc_amount, :so_amount, :sc_amount,
    :lo_status, :lc_status, :so_status, :sc_status,
    :lol_filled, :lcl_filled, :sol_filled, :scl_filled,
    :lo_id, :lc_id, :so_id, :sc_id,
    :lol_pavg, :lcl_pavg, :sol_pavg, :scl_pavg,
    :lastopentrade, :lp_amount, :sp_amount,
]

"""Columns the replay input prepopulates per row; `_rowtakeover!` must never write them."""
const ROWTAKEOVER_PREPOPULATED = Symbol[:opentime, :high, :low, :close, :score, :label]

"""Reference expression of the carry contract, independent of the TSM accessor layer."""
function _rowtakeover_reference!(tdf::DataFrame, ix::Integer)
    ix > 1 || return nothing
    for col in ROWTAKEOVER_CARRIED
        tdf[ix, col] = tdf[ix-1, col]
    end
    return nothing
end

const PROBE_LABELS = [Targets.longopen, Targets.shorthold, Targets.allclose, Targets.shortopen, Targets.longhold]

"""Per-row scores that are distinct and non-zero, so a score carry is detectable."""
probe_scores(n::Int) = Float32[0.5f0 + 0.001f0 * i for i in 1:n]

"""Build a Trades frame whose every cell is unique per (column, row).

Distinct values everywhere are what makes a mis-wired or extra carry detectable; equal
neighbours would silently satisfy the assertions."""
function build_carry_probe(n::Int; scores::Vector{Float32}=probe_scores(n))
    @assert length(scores) == n "scores must cover all $(n) rows; got $(length(scores))"
    df = DataFrame(opentime=[DateTime(2026, 1, 1) + Minute(i) for i in 1:n])
    TSM.ensuretradeschema!(df)
    cols = propertynames(df)
    for i in 1:n
        for (cix, col) in enumerate(cols)
            col === :opentime && continue
            offset = 1000 * cix + i
            if col === :pair
                TSM.settradesfield!(df, i, col, "BTCUSDC")
            elseif col === :score
                TSM.settradesfield!(df, i, col, scores[i])
            elseif col === :label
                TSM.settrades_label!(df, i, PROBE_LABELS[mod1(offset, length(PROBE_LABELS))])
            elseif col === :lastopentrade
                TSM.settradesfield!(df, i, col, DateTime(2020, 1, 1) + Minute(offset))
            elseif col in TSM.TSM_FLOAT_COLUMNS
                TSM.settradesfield!(df, i, col, Float32(offset))
            elseif col in TSM.TSM_INT_COLUMNS
                TSM.settradesfield!(df, i, col, Int32(offset))
            elseif col in TSM.TSM_CATEGORICAL_COLUMNS
                value = if endswith(String(col), "_status")
                    TSM.TSM_STATUS_LEVELS[mod1(offset, length(TSM.TSM_STATUS_LEVELS))]
                elseif col in TSM.TSM_ID_COLUMNS
                    "$(col)-$(offset)"
                else
                    # msg/config/set/tsmstate pools are compressed to UInt8, so cycle a
                    # small level set; adjacent rows still differ, which is what detects carries.
                    "$(col)-$(mod1(offset, 16))"
                end
                TSM.settradesfield!(df, i, col, value)
            else
                error("probe builder does not cover Trades column $(col)::$(eltype(df[!, col]))")
            end
        end
    end
    return df
end

@testset "_rowtakeover! selective carry contract" begin
    @testset "carried fields take the previous row, all others stay untouched" begin
        df = build_carry_probe(3)   # every score is non-zero, matching the replay precondition
        before = deepcopy(df)
        TradingStrategy._rowtakeover!(TSM.TradesColumns(df), 2)

        for col in propertynames(df)
            expected = col in ROWTAKEOVER_CARRIED ? before[1, col] : before[2, col]
            @test isequal(df[2, col], expected)
        end
        # rows other than ix are never touched
        @test all(isequal(df[3, col], before[3, col]) for col in propertynames(df))
    end

    @testset "prepopulated replay columns are never written" begin
        df = build_carry_probe(4)
        before = deepcopy(df)
        cols = TSM.TradesColumns(df)
        for ix in 1:4
            TradingStrategy._rowtakeover!(cols, ix)
        end
        for col in ROWTAKEOVER_PREPOPULATED
            @test isequal(collect(df[!, col]), collect(before[!, col]))
        end
    end

    @testset "ix == 1 is a no-op" begin
        df = build_carry_probe(3)
        before = deepcopy(df)
        TradingStrategy._rowtakeover!(TSM.TradesColumns(df), 1)
        @test all(isequal(df[r, col], before[r, col]) for r in 1:3, col in propertynames(df))
    end

    @testset "matches the reference implementation over a full sequential replay" begin
        n = 400
        production = build_carry_probe(n)
        reference = deepcopy(production)
        productioncols = TSM.TradesColumns(production)

        for ix in 1:n
            TradingStrategy._rowtakeover!(productioncols, ix)
            _rowtakeover_reference!(reference, ix)
        end

        for col in propertynames(production)
            @test isequal(collect(production[!, col]), collect(reference[!, col]))
        end
    end

    @testset "carry set is disjoint from the prepopulated set and within the schema" begin
        schema = propertynames(build_carry_probe(1))
        @test isempty(intersect(ROWTAKEOVER_CARRIED, ROWTAKEOVER_PREPOPULATED))
        @test isempty(setdiff(ROWTAKEOVER_CARRIED, schema))
        @test isempty(setdiff(ROWTAKEOVER_PREPOPULATED, schema))
        @test length(ROWTAKEOVER_CARRIED) == 29
    end
end

function _productsize(rawsize)
    if rawsize isa Tuple
        vals = rawsize
    elseif rawsize isa AbstractVector
        vals = rawsize
    else
        vals = (rawsize,)
    end
    prodsize = 1
    for v in vals
        prodsize *= Int(v)
    end
    return prodsize
end

function _reinterpret_intvector(bytes::Vector{UInt8}, width::Int)
    if width == 1
        return reinterpret(Int8, bytes)
    elseif width == 2
        return reinterpret(Int16, bytes)
    elseif width == 4
        return reinterpret(Int32, bytes)
    elseif width == 8
        return reinterpret(Int64, bytes)
    end
    error("unsupported legacy element width=$(width)")
end

function _repair_legacy_enum_array_payload!(arrdict::Dict{Symbol, Any})::Bool
    if !(get(arrdict, :tag, nothing) == "array")
        return false
    end
    haskey(arrdict, :type) || return false
    haskey(arrdict, :size) || return false
    haskey(arrdict, :data) || return false

    data = arrdict[:data]
    data isa AbstractVector || return false
    bytes = UInt8.(data)

    typ = try
        BSON.raise_recursive(arrdict[:type], @__MODULE__)
    catch
        return false
    end
    typ isa DataType || return false
    isbitstype(typ) || return false
    typ <: Enum || return false

    elementcount = _productsize(arrdict[:size])
    elementcount > 0 || return false

    expectedbytes = elementcount * sizeof(typ)
    length(bytes) == expectedbytes && return false
    (length(bytes) % elementcount == 0) || return false

    oldwidth = length(bytes) ÷ elementcount
    if oldwidth == sizeof(typ)
        return false
    end
    oldwidth in (1, 2, 4, 8) || return false

    rawints = _reinterpret_intvector(bytes, oldwidth)
    bt = Base.Enums.basetype(typ)
    converted = Vector{typ}(undef, length(rawints))
    for i in eachindex(rawints)
        converted[i] = typ(convert(bt, rawints[i]))
    end

    arrdict[:data] = reinterpret(UInt8, converted)
    return true
end

function _repair_legacy_enum_payloads!(obj)::Int
    repaired = 0
    if obj isa Dict{Symbol, Any}
        _repair_legacy_enum_array_payload!(obj) && (repaired += 1)
        for v in values(obj)
            repaired += _repair_legacy_enum_payloads!(v)
        end
    elseif obj isa AbstractVector
        for v in obj
            repaired += _repair_legacy_enum_payloads!(v)
        end
    end
    return repaired
end

function _loadnn_with_legacy_enum_compat(filename::String)
    path = nnfilename(filename)
    raw = BSON.parse(path)
    repaired = _repair_legacy_enum_payloads!(raw)
    repaired > 0 || error("legacy enum payload repair not applicable")

    raised = BSON.raise_recursive(raw, @__MODULE__)
    haskey(raised, :nn) || error("unexpected BSON payload structure: missing :nn key")
    nn = raised[:nn]

    backup = path * ".legacy-enum.bak"
    if !isfile(backup)
        cp(path, backup; force=false)
    end
    BSON.@save path nn
    @warn "loaded legacy classifier artifact via enum-width migration and rewrote file in current format" path backup repaired_arrays=repaired
    return nn
end
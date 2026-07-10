#!/usr/bin/env julia
"""
Script to remove TradeLog-related code from Xch/src/XchCore.jl
"""

const SRCFILE = "Xch/src/XchCore.jl"

# Read the file
lines = readlines(SRCFILE)

# Find functions that reference ExchangeRole or are _tradelog* functions
function_starts = Int[]
function_ends = Int[]
in_function = false
current_start = 0
depth = 0

for (i, line) in enumerate(lines)
    # Skip empty lines and comments
    stripped = strip(line)
    isempty(stripped) && continue
    startswith(stripped, "#") && continue
    
    # Check for function definition
    if startswith(stripped, "function ") || startswith(stripped, "function(")
        if in_function
            # Nested function - should track depth
            depth += 1
        else
            in_function = true
            current_start = i
            depth = 0
        end
        
        # Check if this function should be removed
        if contains(line, "_tradelog") || contains(line, "role::ExchangeRole")
            push!(function_starts, current_start)
        end
    end
    
    # Track 'end' statements
    if stripped == "end"
        if in_function && depth == 0
            in_function = false
            if !isempty(function_starts) && function_starts[end] == current_start
                push!(function_ends, i)
            end
        elseif in_function
            depth -= 1
        end
    end
end

# Also remove TradeLog import
remove_lines = Set{Int}()

# Find and mark TradeLog import line for removal
for (i, line) in enumerate(lines)
    if contains(line, "using") && contains(line, "TradeLog")
        # Mark for removal
        push!(remove_lines, i)
    end
end

# Remove the problematic functions
new_lines = String[]
skip_until = 0

for i in eachindex(lines)
    # Check if we're at the start of a function to remove
    should_remove = false
    for j in eachindex(function_starts)
        if i == function_starts[j]
            should_remove = true
            skip_until = function_ends[j]
            break
        end
    end
    
    if i in remove_lines
        continue  # Skip TradeLog import line
    end
    
    if i > skip_until
        # Add the line back
        push!(new_lines, lines[i])
    elseif i < function_starts[1] || (i > function_ends[end])
        # Before all removals or after all removals
        push!(new_lines, lines[i])
    end
end

# Write back
write(SRCFILE, join(new_lines, "\n") * "\n")
println("Removed TradeLog references from $SRCFILE")

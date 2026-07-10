#!/usr/bin/env python3
"""
Simple script to remove only _tradelog* functions from XchCore.jl
"""

with open("Xch/src/XchCore.jl", "r") as f:
    lines = f.readlines()

# Pass 1: Remove TradeLog from imports
output = []
for line in lines:
    if "using" in line and "TradeLog" in line:
        # Remove TradeLog, keeping other imports
        new_line = line.replace("TradeLog, ", "").replace(", TradeLog", "")
        if new_line.strip() != "using":  # Don't keep empty using
            output.append(new_line)
    else:
        output.append(line)

# Pass 2: Remove all lines between "function _tradelog" and matching "end"
final_output = []
skip_until = -1
in_tradelog_function = False

for i, line in enumerate(final_output):
    if i < skip_until:
        continue
    
    # Check if we're starting a _tradelog function
    if line.strip().startswith("function _tradelog"):
        in_tradelog_function = True
        # Find the matching end
        depth = 1
        for j in range(i + 1, len(output)):
            if output[j].strip().startswith("function "):
                # Nested function
                depth += 1
            elif output[j].strip() == "end":
                depth -= 1
                if depth == 0:
                    skip_until = j + 1
                    break
        continue  # Skip the function line
    
    # Add non-tradelog lines
    final_output.append(line)

# Now construct final output from original output, skipping tradelog functions
final_output = []
i = 0
while i < len(output):
    line = output[i]
    
    if line.strip().startswith("function _tradelog"):
        # Skip this function and all its lines until matching end
        depth = 1
        i += 1
        while i < len(output):
            if output[i].strip().startswith("function "):
                depth += 1
            elif output[i].strip() == "end":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    else:
        final_output.append(line)
        i += 1

# Write back
with open("Xch/src/XchCore.jl", "w") as f:
    f.writelines(final_output)

print(f"Cleaned up: removed _tradelog functions and TradeLog import")
print(f"Original lines: {len(lines)}, Final lines: {len(final_output)}")

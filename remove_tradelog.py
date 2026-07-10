#!/usr/bin/env python3
import re

with open("Xch/src/XchCore.jl", "r") as f:
    lines = f.readlines()

# Process: keep lines but remove _tradelog* and ExchangeRole functions
output = []
skip_until_end = -1
problems = set()

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Check if we should skip (inside a function to remove)
    if i > skip_until_end:
        # Check if this line starts a function to remove
        if re.match(r'^function\s+_tradelog', line):
            # Mark to skip until matching 'end'
            skip_until_end = i
            # Find the matching end
            depth = 0
            for j in range(i, len(lines)):
                if re.match(r'^function\s+', lines[j]):
                    depth += 1
                elif lines[j].strip() == 'end':
                    depth -= 1
                    if depth == 0:
                        skip_until_end = j
                        break
            continue
        
        # Check for functions with ExchangeRole parameter that aren't in our keep list
        if 'role::ExchangeRole' in line and re.match(r'^function\s+', line):
            # These should also be removed
            skip_until_end = i
            depth = 0
            for j in range(i, len(lines)):
                if re.match(r'^function\s+', lines[j]) and j > i:
                    break
                elif lines[j].strip() == 'end':
                    depth -= 1
                    if depth == 0:
                        skip_until_end = j
                        break
            continue
        
        # Skip TradeLog import
        if 'using' in line and 'TradeLog' in line:
            # Remove TradeLog from the using statement
            if ',' in line:
                line = re.sub(r',?\s*TradeLog[,\s]*', '', line)
                line = re.sub(r',\s*,', ',', line)  # Fix double commas
                line = re.sub(r'\s*,\s*\n', '\n', line)
            else:
                continue  # Skip this entire line
        
        output.append(line)

# Write back
with open("Xch/src/XchCore.jl", "w") as f:
    f.writelines(output)

print("Cleaned up TradeLog references")

### Summary

SymbolServer crashes while indexing a minimal environment that has only `Images` as a direct dependency.

Crash headline:

`ERROR: LoadError: (2, Int64)`

### Repro

Project file:

```toml
[deps]
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
```

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=. ~/.vscode/extensions/julialang.language-julia-1.219.2/scripts/packages/SymbolServer/src/server.jl ./symbolstore
```

(Equivalent convenience script: `bash reproduce.sh`)

### Expected

SymbolServer indexes and exits successfully.

### Actual

SymbolServer aborts with stacktrace including:

- `SymbolServer/src/faketypes.jl:40` (`FakeTypeName`)
- `SymbolServer/src/faketypes.jl:149` (`FakeTypeofVararg`)
- `SymbolServer/src/symbols.jl:90` (`DataTypeStore`)

Error top:

```text
ERROR: LoadError: (2, Int64)
Stacktrace:
  [1] error(s::Tuple{Int64, DataType})
  [2] Main.SymbolServer.FakeTypeName(x::Any)
  [3] Main.SymbolServer.FakeTypeofVararg(va::Core.TypeofVararg)
  ...
```

### Environment

- OS: macOS
- Julia: 1.12.6
- VS Code Julia extension: `julialang.language-julia-1.219.2`

### Notes

- This reproduces with a fresh minimal project containing only `Images` in `[deps]`.
- The trigger appears while SymbolServer walks transitive package types.
- I can provide full terminal output and the generated minimal repro folder if needed.

I checked related historical LS/SymbolServer crashes and linked them for context:

https://github.com/julia-vscode/julia-vscode/issues/3451
https://github.com/julia-vscode/julia-vscode/issues/1552
https://github.com/julia-vscode/SymbolServer.jl/issues/150
https://github.com/julia-vscode/julia-vscode/issues/1024
These are all closed/completed, but none appears to match this exact current signature:
ERROR: LoadError: (2, Int64) from FakeTypeName / FakeTypeofVararg (faketypes.jl) while indexing a minimal project with only Images in [deps].

This report includes a fresh minimal repro:

Project.toml
reproduce.sh
README.md


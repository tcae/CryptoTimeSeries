# Images -> SymbolServer crash repro

This is a minimal project that reproduces a Julia VS Code SymbolServer crash with only one direct dependency:

- Images

## Environment used when reproduced

- macOS
- Julia 1.12.6
- VS Code Julia extension: julialang.language-julia-1.219.2
- SymbolServer entrypoint:
  - ~/.vscode/extensions/julialang.language-julia-1.219.2/scripts/packages/SymbolServer/src/server.jl

## Repro steps

1. Run:

   bash reproduce.sh

2. Observe SymbolServer crash while caching Images.

## Expected result

SymbolServer finishes indexing successfully.

## Actual result

SymbolServer aborts with:

ERROR: LoadError: (2, Int64)
Stacktrace:
  [1] error(s::Tuple{Int64, DataType})
    @ Base ./error.jl:54
  [2] Main.SymbolServer.FakeTypeName(x::Any)
    @ .../SymbolServer/src/faketypes.jl:40
  [3] Main.SymbolServer.FakeTypeofVararg(va::Core.TypeofVararg)
    @ .../SymbolServer/src/faketypes.jl:149
  [4] Main.SymbolServer.FakeTypeName(x::Any)
    @ .../SymbolServer/src/faketypes.jl:19
  [5] _parameter(p::Any)
    @ .../SymbolServer/src/faketypes.jl:70
  ...

## Notes

- This repro uses only Images as a direct dependency in Project.toml.
- RegionTrees and ImageSegmentation are pulled transitively by Images in the resolved manifest.
- If the extension path differs on your machine, set JULIA_SYMBOLSERVER_SERVER_JL before running:

  JULIA_SYMBOLSERVER_SERVER_JL=/full/path/to/server.jl bash reproduce.sh

## Issue templates

- SymbolServer issue template: `issue-body-symbolserver.md`
- Legacy generic issue template: `issue-body.md`
- Historical references and status: `references-and-status.md`

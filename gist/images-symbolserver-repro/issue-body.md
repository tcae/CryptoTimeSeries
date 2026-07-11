### Summary

When SymbolServer indexes a project whose only direct dependency is `Images`, indexing crashes with:

`ERROR: LoadError: (2, Int64)`

The failure happens in `FakeTypeName` / `FakeTypeofVararg` inside SymbolServer as used by the VS Code Julia extension.

### Minimal project

Use these files:

- `Project.toml` with only:

```toml
[deps]
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
```

- run command (or `bash reproduce.sh`):

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=. ~/.vscode/extensions/julialang.language-julia-1.219.2/scripts/packages/SymbolServer/src/server.jl ./symbolstore
```

### Expected

SymbolServer indexes successfully.

### Actual

SymbolServer crashes while caching `Images` with stacktrace starting at:

- `SymbolServer/src/faketypes.jl:40`
- `SymbolServer/src/faketypes.jl:149`
- `SymbolServer/src/symbols.jl:90`

Error headline:

`ERROR: LoadError: (2, Int64)`

### Environment

- OS: macOS
- Julia: 1.12.6
- VS Code Julia extension: `julialang.language-julia-1.219.2`

### Notes

`Project.toml` has only `Images` as a direct dependency, and the failure occurs in SymbolServer's type-walk path during indexing.

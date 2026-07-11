# Related Issues And Status

This note collects previously reported, related Language Server / SymbolServer crashes and their current GitHub state.

## Closest related reports

1. https://github.com/julia-vscode/julia-vscode/issues/3451
- Title: the language server can't start with latest Julia nightly
- Similarity: same SymbolServer `FakeTypeName` crash area in `faketypes.jl`
- Current state: Closed (completed)
- Resolution confidence for this exact bug: Low (different payload and nightly-specific context)

2. https://github.com/julia-vscode/julia-vscode/issues/1552
- Title: Julia Language Server server crashed in Julia 1.6.0-DEV
- Similarity: `FakeTypeName` crash path in SymbolServer
- Current state: Closed (completed)
- Resolution confidence for this exact bug: Low (older versions and different error tuple)

3. https://github.com/julia-vscode/SymbolServer.jl/issues/150
- Title: ERROR: LoadError: type UnionAll has no field name
- Similarity: same `faketypes.jl` family of failures
- Current state: Closed (completed)
- Resolution confidence for this exact bug: Low (different concrete failure)

4. https://github.com/julia-vscode/julia-vscode/issues/1024
- Title: Failed to precompile LanguageServer
- Similarity: broad LS/SymbolServer precompile failures; includes historical `TypeofVararg` mention in comments
- Current state: Closed (completed)
- Resolution confidence for this exact bug: Low (mixed causes, many old-extension reports)

## Are the other issues reported as resolved?

Short answer: they are marked resolved on GitHub (closed/completed), but they do not look like an exact duplicate of the current `(2, Int64)` crash on `julia-vscode 1.219.2` + Julia `1.12.6`.

Practical interpretation:
- Those issues indicate this crash family has happened before.
- Your current signature appears new enough to justify a new SymbolServer issue with a fresh MWE.

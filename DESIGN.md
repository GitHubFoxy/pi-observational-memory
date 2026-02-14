# Observational Memory for pi (Design Notes)

This file captures the implementation plan and conventions so the extension can be maintained without digging through chat history.

## Constraints

- No pi source modification or fork
- Implement through extension hooks only
- Use current active model for compaction generation (can be swapped later)

## Extension style followed (from pi examples)

- `index.ts` as default extension entry
- Top-level JSDoc block describing behavior/usage
- Helpers as pure functions outside hook handlers
- Defensive fallbacks (return `undefined` to allow default behavior)
- Optional local package metadata + lint config (`package.json`, `biome.json`)
- Project docs colocated with extension (`README.md`, this file)

## Implemented (v3)

- `session_before_compact` override:
  - Reads `preparation.messagesToSummarize`, `turnPrefixMessages`, `previousSummary`
  - Serializes conversation via `convertToLlm` + `serializeConversation`
  - Generates observation summary with active model
  - Returns custom `compaction` result
- Reflector pass:
  - Dedupes observations
  - Priority-aware pruning with caps (🔴/🟡/🟢)
  - Triggered by observation-token threshold (default 40k) or forced mode
- `session_before_tree` override:
  - Uses `prepareBranchEntries()` to gather branch messages/file ops
  - Generates observational branch summaries
- Observer auto-compaction trigger:
  - Runs on `agent_end` in buffered mode
  - Estimates raw-tail token block from session entries after latest compaction
  - Triggers compaction at configurable activation threshold: observer + retain buffer (defaults: 30k + 8k)
- Observer modes:
  - `buffered` (default): background trigger on `agent_end`
  - `blocking`: disable background observer trigger (manual/regular compaction still active)
- Two-threshold OM flow:
  - Observer trigger: raw-tail tokens (default 30k)
  - Reflector trigger: observation-block tokens (default 40k)
- Partial activation:
  - Raw-tail retain buffer (default 8k) to preserve more recent raw history before observer compaction fires
- Summary format:
  - `## Observations`
  - `## Open Threads`
  - `## Next Action Bias`
- File operation tags:
  - `<read-files>` and `<modified-files>`
  - merged cumulatively with previous compaction tags
- Commands:
  - `/obs-memory-status`
  - `/obs-auto-compact [on|off] [mode] [observerTokens] [reflectorTokens] [retainTokens]`
  - `/obs-mode [buffered|blocking]`
  - `/obs-view [obs] [raw] [maxLines]`
  - `/obs-reflect [extra focus]`

## Next milestones

1. Add maintenance commands:
   - `/obs-rebuild`
   - `/obs-dump`
2. Persist runtime mode/threshold overrides across reloads/sessions
3. Add richer UI viewer (scrollable popover) for observations

## Operational notes

- If no model/API key, extension intentionally falls back to built-in compaction
- `firstKeptEntryId` always comes from `preparation` to preserve pi semantics
- `details.schemaVersion` guards forward compatibility

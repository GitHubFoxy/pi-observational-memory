# Observational Memory Extension

Extension-only custom compaction for pi, inspired by observational-memory systems.

See [DESIGN.md](./DESIGN.md) for architecture notes and next milestones.

This implementation follows pi's extension style:
- directory extension with `index.ts`
- top-level JSDoc usage/behavior comments
- explicit command registration
- fallback-safe hooks (default compaction remains available)

## What it does now

On `session_before_compact`:
1. Uses pi's prepared compaction input (`messagesToSummarize`, `turnPrefixMessages`, `previousSummary`)
2. Uses the **current active model** (same model as your session)
3. Generates a structured summary with sections:
   - `## Observations`
   - `## Open Threads`
   - `## Next Action Bias`
4. Runs reflector GC when observation block token estimate crosses threshold (default: 40k) or when forced via command
5. Preserves pi's kept-tail behavior by reusing `preparation.firstKeptEntryId`
6. Appends cumulative `<read-files>` and `<modified-files>` tags (merged with prior checkpoint tags)
7. Stores extension metadata in `compaction.details`

On `session_before_tree`:
- Replaces default branch summary with the same observational format.

On `agent_end` (buffered mode):
- Estimates raw-tail tokens (messages since latest compaction) and triggers observer compaction at threshold (default: 30k + 8k retain buffer).

Observer modes:
- `buffered` (default): auto observer checks in background after agent turns.
- `blocking`: background observer trigger is disabled; regular/manual compaction still works.

If generation fails or API key/model is unavailable, it returns nothing so pi falls back to default compaction/tree summarization.

## Commands

- `/obs-memory-status` — show latest compaction + branch summary metadata and OM token block estimates
- `/obs-auto-compact [on|off] [mode] [observerTokens] [reflectorTokens] [retainTokens]` — show/set thresholds and mode
  - keyed form: `/obs-auto-compact mode=buffered observer=30k reflector=40k retain=8k`
- `/obs-mode [buffered|blocking]` — show/set observer auto-compaction mode
- `/obs-view [obs] [raw] [maxLines]` — inspect latest observation summary quickly in-terminal
- `/obs-reflect [extra focus]` — force aggressive reflection on next compaction and trigger compaction now

Optional startup flags:
- `--obs-auto-compact=true|false`
- `--obs-mode=buffered|blocking`
- `--obs-observer-threshold=30000` (or `30k`)
- `--obs-reflector-threshold=40000` (or `40k`)
- `--obs-retain-raw-tail=8000` (or `8k`, `0` to disable retain buffer)

## Install

From npm (recommended):

```bash
pi install npm:pi-extension-observational-memory
```

Or add to settings directly:

```json
{
  "packages": ["npm:pi-extension-observational-memory"]
}
```

For local development, you can still load from disk:

- global: `~/.pi/agent/extensions/observational-memory/index.ts`
- project: `.pi/extensions/observational-memory/index.ts`

Or run one-off:

```bash
pi -e ./index.ts
```

## Linting / formatting

This folder includes a local `package.json` + `biome.json` so it can be checked independently.

```bash
cd path/to/pi-extension-observational-memory
npm install
npm run lint
npm run format
```

## Current scope vs target design

Implemented now:
- Observer-style compaction summary override
- Prior-summary carry forward
- Reflector/GC pass (threshold + forced mode)
- Branch-aware observational summaries via `session_before_tree`
- Extension-managed two-threshold OM flow: observer trigger (default 30k) + reflector trigger (default 40k)
- Buffered/background observer mode by default, with optional blocking mode
- Partial activation support via raw-tail retain buffer (default 8k)
- Commands: `/obs-memory-status`, `/obs-auto-compact`, `/obs-mode`, `/obs-view`, `/obs-reflect`

Planned next:
- Additional maintenance commands (`/obs-rebuild`, `/obs-dump`)

## Notes

- This extension intentionally does **not** change pi core internals.
- It composes with pi's existing session tree and compaction lifecycle.
- Model choice is currently the active session model; can be changed later if desired.

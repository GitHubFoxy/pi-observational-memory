/**
 * Observational Memory Extension
 *
 * Extension-only memory strategy for pi compaction + tree summarization.
 *
 * Core behavior:
 * - Overrides `session_before_compact` with observational summaries.
 * - Uses the active session model (parent model) by default.
 * - Runs a lightweight reflector pass (dedupe/prune) on large observation sets.
 * - Optionally forces aggressive reflection via `/obs-reflect`.
 * - Overrides `session_before_tree` summaries with the same observational format.
 * - Triggers observer auto-compaction at configurable raw-tail tokens (default 30k).
 * - Supports buffered (default) and blocking observer modes.
 * - Supports partial activation via raw-tail retention buffer (default 8k).
 * - Triggers reflector GC at configurable observation-block tokens (default 40k).
 */

import { completeSimple, type Model } from "@mariozechner/pi-ai";
import {
	type CompactionResult,
	convertToLlm,
	type ExtensionAPI,
	type ExtensionContext,
	estimateTokens,
	type FileOperations,
	prepareBranchEntries,
	type SessionEntry,
	serializeConversation,
} from "@mariozechner/pi-coding-agent";
import { ObservationMemoryOverlay, type ObservationMemoryOverlaySnapshot } from "./overlay.js";

const DETAILS_SCHEMA_VERSION = 2;

const OBS_STATUS_COMMAND = "obs-memory-status";
const OBS_REFLECT_COMMAND = "obs-reflect";
const OBS_AUTO_COMPACT_COMMAND = "obs-auto-compact";
const OBS_MODE_COMMAND = "obs-mode";
const OBS_VIEW_COMMAND = "obs-view";
const OBS_STATUS_SHORTCUT = "ctrl+shift+o";

const DEFAULT_RESERVE_TOKENS = 16384;

const OBS_SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant for a coding agent.
Produce concise markdown summaries only.
Use only information explicitly present in the provided conversation context.
If information is missing, use "unknown" rather than guessing.
Never call tools.
Follow the user's format instructions exactly.`;

const DEFAULT_OBS_MODE = "buffered" as const;
const DEFAULT_OBSERVER_TRIGGER_TOKENS = 30_000;
const DEFAULT_REFLECTOR_TRIGGER_TOKENS = 40_000;
const DEFAULT_RAW_TAIL_RETAIN_TOKENS = 8_000;
const AUTO_COMPACT_COOLDOWN_MS = 5000;
const AUTO_TOKENS_MIN = 2_000;
const AUTO_TOKENS_MAX = 500_000;

const REFLECT_LIMITS_THRESHOLD = {
	red: 96,
	yellow: 40,
	green: 16,
} as const;

const REFLECT_LIMITS_FORCED = {
	red: 72,
	yellow: 28,
	green: 8,
} as const;

type ReflectionMode = "none" | "threshold" | "forced";
type ObservationPriority = "red" | "yellow" | "green";
type AutoCompactionMode = "buffered" | "blocking";

interface ObservationalCompactionDetails {
	schemaVersion: number;
	strategy: "observational-memory";
	model: string;
	observationCount: number;
	observationCountBefore: number;
	observationCountAfter: number;
	observationsDropped: number;
	reflectorRan: boolean;
	reflectionMode: ReflectionMode;
	generatedAt: string;
	isSplitTurn: boolean;
	usedPreviousSummary: boolean;
}

interface ObservationalBranchSummaryDetails {
	schemaVersion: number;
	strategy: "observational-memory-tree";
	model: string;
	observationCount: number;
	generatedAt: string;
	entryCount: number;
}

interface ParsedObservation {
	priority: ObservationPriority;
	body: string;
	key: string;
	index: number;
}

interface ReflectionResult {
	summary: string;
	before: number;
	after: number;
	dropped: number;
}

function buildModelRef(model: Model<any>): string {
	return `${model.provider}/${model.id}`;
}

function normalizeSummary(raw: string): string {
	const text = raw.trim();
	if (text.length === 0) {
		return [
			"## Observations",
			"Date: unknown",
			"- 🟡 Unable to extract observations from compaction source.",
			"",
			"## Open Threads",
			"- Continue from retained recent context.",
			"",
			"## Next Action Bias",
			"1. Continue the latest user request from recent raw context.",
		].join("\n");
	}

	if (text.includes("## Observations") && text.includes("## Open Threads") && text.includes("## Next Action Bias")) {
		return text;
	}

	return [
		"## Observations",
		"Date: unknown",
		"- 🟡 Model returned non-standard output; preserving raw output below.",
		"",
		"## Open Threads",
		"- Continue from retained recent context.",
		"",
		"## Next Action Bias",
		"1. Continue the latest user request from recent raw context.",
		"",
		"## Raw Observer Output",
		text,
	].join("\n");
}

function countObservationLines(summary: string): number {
	const matches = summary.match(/^\s*-\s*[🔴🟡🟢]\s+/gmu);
	return matches?.length ?? 0;
}

function estimateTextTokens(text: string): number {
	if (!text.trim()) return 0;
	return Math.ceil(text.length / 4);
}

function formatTokenCount(tokens: number): string {
	return `${tokens.toLocaleString()} tokens`;
}

function parseTokenCount(
	input: string,
	options: {
		min?: number;
		max?: number;
		allowZero?: boolean;
	} = {},
): number | undefined {
	const min = options.min ?? AUTO_TOKENS_MIN;
	const max = options.max ?? AUTO_TOKENS_MAX;
	const allowZero = options.allowZero ?? false;

	const raw = input.trim().toLowerCase().replaceAll(",", "").replaceAll("_", "");
	if (raw.length === 0) return undefined;

	let multiplier = 1;
	let numericPart = raw;
	if (raw.endsWith("k")) {
		multiplier = 1_000;
		numericPart = raw.slice(0, -1);
	} else if (raw.endsWith("m")) {
		multiplier = 1_000_000;
		numericPart = raw.slice(0, -1);
	}

	if (numericPart.length === 0) return undefined;
	const numeric = Number(numericPart);
	if (!Number.isFinite(numeric) || numeric < 0) {
		return undefined;
	}

	const tokens = Math.round(numeric * multiplier);
	if (!allowZero && tokens <= 0) {
		return undefined;
	}
	if (allowZero && tokens === 0) {
		return 0;
	}
	if (tokens < min || tokens > max) {
		return undefined;
	}

	return tokens;
}

function parseAutoCompactionMode(token: string): AutoCompactionMode | undefined {
	const normalized = token.trim().toLowerCase();
	if (["buffered", "buffer", "bg", "background", "async", "auto"].includes(normalized)) {
		return "buffered";
	}
	if (["blocking", "sync", "manual"].includes(normalized)) {
		return "blocking";
	}
	return undefined;
}

function parseEnabledToken(token: string): boolean | undefined {
	const normalized = token.trim().toLowerCase();
	if (["on", "enable", "enabled", "true", "1"].includes(normalized)) return true;
	if (["off", "disable", "disabled", "false", "0"].includes(normalized)) return false;
	return undefined;
}

function parseRetainRawTailTokenCount(token: string): number | undefined {
	const normalized = token.trim().toLowerCase();
	if (["off", "none", "disable", "disabled", "false"].includes(normalized)) return 0;
	return parseTokenCount(token, {
		min: 0,
		max: AUTO_TOKENS_MAX,
		allowZero: true,
	});
}

function shouldIgnoreAutoCompactError(message: string): boolean {
	return /nothing to compact|already compacted|compaction cancelled/i.test(message);
}

function estimateCustomMessageEntryTokens(entry: Extract<SessionEntry, { type: "custom_message" }>): number {
	let chars = 0;
	if (typeof entry.content === "string") {
		chars = entry.content.length;
	} else {
		for (const block of entry.content) {
			if (block.type === "text") {
				chars += block.text.length;
			} else if (block.type === "image") {
				chars += 4800;
			}
		}
	}
	return Math.ceil(chars / 4);
}

function estimateObservationTokens(summary: string | undefined): number {
	if (!summary) return 0;
	return estimateTextTokens(stripFileTags(summary));
}

function getObserverActivationThreshold(observerTriggerTokens: number, rawTailRetainTokens: number): number {
	return observerTriggerTokens + rawTailRetainTokens;
}

function estimateRawTailTokens(branchEntries: SessionEntry[]): number {
	let startIndex = 0;
	for (let i = branchEntries.length - 1; i >= 0; i--) {
		if (branchEntries[i].type === "compaction") {
			startIndex = i + 1;
			break;
		}
	}

	let totalTokens = 0;
	for (let i = startIndex; i < branchEntries.length; i++) {
		const entry = branchEntries[i];
		switch (entry.type) {
			case "message":
				totalTokens += estimateTokens(entry.message);
				break;
			case "custom_message":
				totalTokens += estimateCustomMessageEntryTokens(entry);
				break;
			case "branch_summary":
				totalTokens += estimateTextTokens(stripFileTags(entry.summary));
				break;
			default:
				break;
		}
	}

	return totalTokens;
}

function parseTaggedFiles(summary: string | undefined, tag: "read-files" | "modified-files"): Set<string> {
	if (!summary) return new Set<string>();
	const regex = new RegExp(`<${tag}>\\n([\\s\\S]*?)\\n<\\/${tag}>`, "m");
	const match = summary.match(regex);
	if (!match?.[1]) return new Set<string>();
	return new Set(
		match[1]
			.split("\n")
			.map((line) => line.trim())
			.filter((line) => line.length > 0),
	);
}

function stripFileTags(summary: string): string {
	return summary
		.replace(/\n?<read-files>\n[\s\S]*?\n<\/read-files>/gm, "")
		.replace(/\n?<modified-files>\n[\s\S]*?\n<\/modified-files>/gm, "")
		.trim();
}

function formatFileOperations(fileOps: FileOperations, previousSummary?: string): string {
	const previousRead = parseTaggedFiles(previousSummary, "read-files");
	const previousModified = parseTaggedFiles(previousSummary, "modified-files");

	const currentModified = new Set<string>([...fileOps.edited, ...fileOps.written]);
	const currentRead = new Set<string>([...fileOps.read].filter((filePath) => !currentModified.has(filePath)));

	const modified = new Set<string>([...previousModified, ...currentModified]);
	const read = new Set<string>([...previousRead, ...currentRead]);
	for (const filePath of modified) {
		read.delete(filePath);
	}

	const readFiles = [...read].sort();
	const modifiedFiles = [...modified].sort();

	const sections: string[] = [];
	if (readFiles.length > 0) {
		sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
	}
	if (modifiedFiles.length > 0) {
		sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
	}
	if (sections.length === 0) return "";
	return `\n\n${sections.join("\n\n")}`;
}

function extractSection(summary: string, heading: string, nextHeading?: string): string {
	const start = summary.indexOf(heading);
	if (start === -1) return "";

	let content = summary.slice(start + heading.length);
	content = content.replace(/^\s*\n/, "");

	if (!nextHeading) {
		return content.trim();
	}

	const end = content.indexOf(nextHeading);
	if (end === -1) {
		return content.trim();
	}
	return content.slice(0, end).trim();
}

function priorityFromEmoji(emoji: string): ObservationPriority {
	switch (emoji) {
		case "🔴":
			return "red";
		case "🟡":
			return "yellow";
		default:
			return "green";
	}
}

function priorityRank(priority: ObservationPriority): number {
	switch (priority) {
		case "red":
			return 3;
		case "yellow":
			return 2;
		default:
			return 1;
	}
}

function normalizeObservationKey(body: string): string {
	return body
		.replace(/^\d{1,2}:\d{2}\s+/, "")
		.toLowerCase()
		.replace(/[`*_~()[\]{}<>]/g, "")
		.replace(/\s+/g, " ")
		.trim();
}

function parseObservationLines(observationsSection: string): ParsedObservation[] {
	const lines = observationsSection.split("\n");
	const parsed: ParsedObservation[] = [];

	for (let index = 0; index < lines.length; index++) {
		const line = lines[index].trim();
		const match = line.match(/^\s*-\s*(🔴|🟡|🟢)\s+(.+)$/u);
		if (!match) continue;

		const body = match[2].trim();
		if (!body) continue;
		const key = normalizeObservationKey(body);
		if (!key) continue;

		parsed.push({
			priority: priorityFromEmoji(match[1]),
			body,
			key,
			index,
		});
	}

	return parsed;
}

function dedupeAndLimitObservations(observations: ParsedObservation[], forced: boolean): ParsedObservation[] {
	const byKey = new Map<string, ParsedObservation>();

	for (const item of observations) {
		const previous = byKey.get(item.key);
		if (!previous) {
			byKey.set(item.key, item);
			continue;
		}

		const previousRank = priorityRank(previous.priority);
		const currentRank = priorityRank(item.priority);
		if (currentRank > previousRank) {
			byKey.set(item.key, item);
			continue;
		}

		if (currentRank === previousRank && item.index > previous.index) {
			byKey.set(item.key, item);
		}
	}

	const unique = [...byKey.values()].sort((a, b) => {
		const rankDelta = priorityRank(b.priority) - priorityRank(a.priority);
		if (rankDelta !== 0) return rankDelta;
		return b.index - a.index;
	});

	const limits = forced ? REFLECT_LIMITS_FORCED : REFLECT_LIMITS_THRESHOLD;
	const picked: ParsedObservation[] = [];
	const counts = { red: 0, yellow: 0, green: 0 };

	for (const item of unique) {
		if (counts[item.priority] >= limits[item.priority]) continue;
		counts[item.priority]++;
		picked.push(item);
	}

	return picked.sort((a, b) => {
		const rankDelta = priorityRank(b.priority) - priorityRank(a.priority);
		if (rankDelta !== 0) return rankDelta;
		return b.index - a.index;
	});
}

function dedupeTextLines(lines: string[], maxItems: number): string[] {
	const seen = new Set<string>();
	const output: string[] = [];

	for (const line of lines) {
		const normalized = line.toLowerCase().replace(/\s+/g, " ").trim();
		if (!normalized || seen.has(normalized)) continue;
		seen.add(normalized);
		output.push(line.trim());
		if (output.length >= maxItems) break;
	}

	return output;
}

function reflectSummary(summary: string, mode: ReflectionMode): ReflectionResult {
	if (mode === "none") {
		const count = countObservationLines(summary);
		return {
			summary,
			before: count,
			after: count,
			dropped: 0,
		};
	}

	const observationsSection = extractSection(summary, "## Observations", "## Open Threads");
	const openThreadsSection = extractSection(summary, "## Open Threads", "## Next Action Bias");
	const nextActionSection = extractSection(summary, "## Next Action Bias");

	const parsedObservations = parseObservationLines(observationsSection);
	const reflectedObservations = dedupeAndLimitObservations(parsedObservations, mode === "forced");

	const openThreadLines = dedupeTextLines(
		openThreadsSection
			.split("\n")
			.map((line) => line.trim())
			.filter((line) => line.startsWith("- "))
			.map((line) => line.replace(/^-\s+/, "").trim()),
		12,
	);

	const nextActionLines = dedupeTextLines(
		nextActionSection
			.split("\n")
			.map((line) => line.trim())
			.filter((line) => /^\d+\.\s+/.test(line))
			.map((line) => line.replace(/^\d+\.\s+/, "").trim()),
		4,
	);

	const reflected = [
		"## Observations",
		"Date: reflected",
		...(reflectedObservations.length > 0
			? reflectedObservations.map((item) => {
					const emoji = item.priority === "red" ? "🔴" : item.priority === "yellow" ? "🟡" : "🟢";
					return `- ${emoji} ${item.body}`;
				})
			: ["- 🟡 No durable observations extracted."]),
		"",
		"## Open Threads",
		...(openThreadLines.length > 0 ? openThreadLines.map((line) => `- ${line}`) : ["- (none)"]),
		"",
		"## Next Action Bias",
		...(nextActionLines.length > 0
			? nextActionLines.map((line, index) => `${index + 1}. ${line}`)
			: ["1. Continue from the latest user request and retained recent context."]),
	].join("\n");

	return {
		summary: reflected,
		before: parsedObservations.length,
		after: reflectedObservations.length,
		dropped: Math.max(0, parsedObservations.length - reflectedObservations.length),
	};
}

function buildCompactionPrompt(
	conversationText: string,
	options: {
		previousSummary?: string;
		customInstructions?: string;
		isSplitTurn: boolean;
		forceReflect: boolean;
	},
): string {
	const previousSummaryBlock = options.previousSummary
		? `<previous-observations>\n${options.previousSummary}\n</previous-observations>\n\n`
		: "";

	const splitTurnNote = options.isSplitTurn
		? "NOTE: This compaction split a large turn. Keep enough context to understand the retained recent suffix."
		: "";

	const forceReflectNote = options.forceReflect
		? "FORCED REFLECTOR MODE: aggressively deduplicate observations and prune stale low-priority context."
		: "";

	const customInstructionsBlock = options.customInstructions
		? `\n\nAdditional focus from user:\n${options.customInstructions}`
		: "";

	return `You are an observational memory compressor for a coding agent.

Your job:
- Convert conversation history into durable observation logs.
- Keep facts/constraints/decisions needed for future work.
- Prefer concise, high-signal lines.
- Preserve critical names, file paths, APIs, errors, and deadlines.
- Preserve useful previous observations unless contradicted.

Rules:
1) Output ONLY markdown in the exact section structure below.
2) Use emoji priorities per line:
   - 🔴 critical constraints, blockers, deadlines, irreversible decisions
   - 🟡 important but possibly evolving context
   - 🟢 low-priority informational context
3) Every bullet must be grounded in the provided conversation or previous observations. Never invent file names, commands, errors, dates, or timestamps.
4) If exact dates/times are not explicitly present, use "Date: unknown" and omit HH:mm prefixes.
5) Keep each bullet single-line and concrete.
6) Do not answer the user. Do not continue the conversation.

Required output format:

## Observations
Date: unknown
- 🔴 [observation]
- 🟡 [observation]
- 🟢 [observation]

## Open Threads
- [unfinished work item]
- [or "(none)"]

## Next Action Bias
1. [most likely immediate next action]
2. [optional second action]

${splitTurnNote}
${forceReflectNote}

${previousSummaryBlock}<conversation>
${conversationText}
</conversation>${customInstructionsBlock}`;
}

function buildTreePrompt(
	conversationText: string,
	options: {
		customInstructions?: string;
		replaceInstructions?: boolean;
	},
): string {
	const customInstructionsBlock = options.customInstructions
		? options.replaceInstructions
			? `\n\nCustom summarization instructions (replace mode, highest priority):\n${options.customInstructions}`
			: `\n\nAdditional focus from user:\n${options.customInstructions}`
		: "";

	return `You are summarizing an abandoned branch from a coding-agent session.

Produce an observational-memory style summary so the active branch can retain important context.

Rules:
- Keep only durable, actionable context.
- Preserve critical file paths, decisions, blockers, and requirements.
- Use priorities: 🔴 critical, 🟡 important, 🟢 informational.
- Every bullet must be grounded in the provided conversation. Never invent file names, commands, errors, dates, or timestamps.
- If exact dates/times are not explicitly present, use "Date: unknown" and omit HH:mm prefixes.
- Output ONLY markdown in this exact structure:

## Observations
Date: unknown
- 🔴 [observation]
- 🟡 [observation]
- 🟢 [observation]

## Open Threads
- [unfinished work item]
- [or "(none)"]

## Next Action Bias
1. [most likely immediate next action]
2. [optional second action]

<conversation>
${conversationText}
</conversation>${customInstructionsBlock}`;
}

async function summarizeWithModel(
	model: Model<any>,
	apiKey: string,
	promptText: string,
	maxTokens: number,
	signal: AbortSignal,
): Promise<string> {
	const runSummarization = async (inputPrompt: string): Promise<string> => {
		const response = await completeSimple(
			model,
			{
				systemPrompt: OBS_SUMMARIZATION_SYSTEM_PROMPT,
				messages: [
					{
						role: "user",
						content: [{ type: "text", text: inputPrompt }],
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey, maxTokens, signal, reasoning: "high" },
		);

		if (response.stopReason === "error") {
			throw new Error(`Summarization failed: ${response.errorMessage || "Unknown error"}`);
		}
		if (response.stopReason === "aborted") {
			throw new Error("Summarization aborted");
		}
		if (response.stopReason === "toolUse") {
			throw new Error("Summarization unexpectedly requested tool use");
		}

		return response.content
			.filter((part): part is { type: "text"; text: string } => part.type === "text")
			.map((part) => part.text)
			.join("\n")
			.trim();
	};

	const firstAttempt = await runSummarization(promptText);
	if (firstAttempt.length > 0) {
		return firstAttempt;
	}

	const retryPrompt = `${promptText}\n\nIMPORTANT: Output ONLY markdown in the required three-section format. Do not return empty output.`;
	const secondAttempt = await runSummarization(retryPrompt);
	if (secondAttempt.length > 0) {
		return secondAttempt;
	}

	throw new Error("Summarization returned empty text");
}

function isObservationalCompactionDetails(value: unknown): value is ObservationalCompactionDetails {
	if (!value || typeof value !== "object") return false;
	const maybe = value as Record<string, unknown>;
	return (
		typeof maybe.schemaVersion === "number" &&
		maybe.strategy === "observational-memory" &&
		typeof maybe.model === "string" &&
		typeof maybe.observationCount === "number"
	);
}

function isObservationalBranchDetails(value: unknown): value is ObservationalBranchSummaryDetails {
	if (!value || typeof value !== "object") return false;
	const maybe = value as Record<string, unknown>;
	return (
		typeof maybe.schemaVersion === "number" &&
		maybe.strategy === "observational-memory-tree" &&
		typeof maybe.model === "string" &&
		typeof maybe.observationCount === "number"
	);
}

export default function observationalMemoryExtension(pi: ExtensionAPI) {
	let forceReflectNextCompaction = false;
	let autoObserverEnabled = true;
	let autoCompactionMode: AutoCompactionMode = DEFAULT_OBS_MODE;
	let observerTriggerTokens = DEFAULT_OBSERVER_TRIGGER_TOKENS;
	let reflectorTriggerTokens = DEFAULT_REFLECTOR_TRIGGER_TOKENS;
	let rawTailRetainTokens = DEFAULT_RAW_TAIL_RETAIN_TOKENS;
	let autoCompactInFlight = false;
	let lastAutoCompactAt = 0;
	let statusOverlayOpen = false;

	pi.registerFlag("obs-auto-compact", {
		description: "Enable observational auto observer trigger",
		type: "boolean",
		default: true,
	});

	pi.registerFlag("obs-mode", {
		description: "Auto-compaction mode: buffered (background) or blocking",
		type: "string",
		default: DEFAULT_OBS_MODE,
	});

	pi.registerFlag("obs-observer-threshold", {
		description: "Observer trigger threshold for raw-tail tokens (e.g. 30000 or 30k)",
		type: "string",
		default: String(DEFAULT_OBSERVER_TRIGGER_TOKENS),
	});

	pi.registerFlag("obs-reflector-threshold", {
		description: "Reflector trigger threshold for observation-block tokens (e.g. 40000 or 40k)",
		type: "string",
		default: String(DEFAULT_REFLECTOR_TRIGGER_TOKENS),
	});

	pi.registerFlag("obs-retain-raw-tail", {
		description: "Extra raw-tail tokens to retain before observer compaction triggers (e.g. 8000 or 8k)",
		type: "string",
		default: String(DEFAULT_RAW_TAIL_RETAIN_TOKENS),
	});

	const triggerObserverAutoCompaction = (ctx: ExtensionContext, options?: { forced?: boolean }) => {
		const forced = options?.forced ?? false;
		if ((!autoObserverEnabled && !forced) || autoCompactInFlight) return;
		if (autoCompactionMode !== "buffered" && !forced) return;

		const now = Date.now();
		if (now - lastAutoCompactAt < AUTO_COMPACT_COOLDOWN_MS) return;

		const branchEntries = ctx.sessionManager.getBranch();
		const rawTailTokens = estimateRawTailTokens(branchEntries);
		const activationThreshold = getObserverActivationThreshold(observerTriggerTokens, rawTailRetainTokens);
		if (rawTailTokens < activationThreshold) return;

		const lastCompaction = [...branchEntries].reverse().find((entry) => entry.type === "compaction");
		const observationTokens = estimateObservationTokens(lastCompaction?.summary);

		autoCompactInFlight = true;
		lastAutoCompactAt = now;

		if (ctx.hasUI) {
			ctx.ui.notify(
				[
					"Obs observer trigger reached:",
					`- mode: ${autoCompactionMode}`,
					`- raw tail: ${formatTokenCount(rawTailTokens)}`,
					`- observer threshold: ${formatTokenCount(observerTriggerTokens)}`,
					`- raw-tail retain: ${formatTokenCount(rawTailRetainTokens)}`,
					`- activation threshold: ${formatTokenCount(activationThreshold)}`,
					`- observation block: ${formatTokenCount(observationTokens)}`,
					`- reflector threshold: ${formatTokenCount(reflectorTriggerTokens)}`,
				].join("\n"),
				"info",
			);
		}

		ctx.compact({
			customInstructions: [
				`Observer trigger fired because raw tail reached ${formatTokenCount(rawTailTokens)} (activation threshold ${formatTokenCount(activationThreshold)}).`,
				`Observer threshold=${formatTokenCount(observerTriggerTokens)}, raw-tail retain=${formatTokenCount(rawTailRetainTokens)}, mode=${autoCompactionMode}.`,
				`Current observation block estimate: ${formatTokenCount(observationTokens)} (reflector threshold ${formatTokenCount(reflectorTriggerTokens)}).`,
				"Preserve critical constraints, blockers, decisions, active tasks, and the latest relevant context from the current raw tail.",
			].join("\n"),
			onComplete: (result) => {
				autoCompactInFlight = false;
				if (ctx.hasUI) {
					ctx.ui.notify(
						`Obs observer compaction complete (${result.tokensBefore.toLocaleString()} tokens before).`,
						"info",
					);
				}
			},
			onError: (error) => {
				autoCompactInFlight = false;
				const message = error instanceof Error ? error.message : String(error);
				if (ctx.hasUI && !shouldIgnoreAutoCompactError(message)) {
					ctx.ui.notify(`Obs observer compaction failed: ${message}`, "error");
				}
			},
		});
	};

	const buildStatusSnapshot = (ctx: ExtensionContext): ObservationMemoryOverlaySnapshot => {
		const branchEntries = ctx.sessionManager.getBranch();
		const lastCompaction = [...branchEntries].reverse().find((entry) => entry.type === "compaction");
		const lastBranchSummary = [...branchEntries].reverse().find((entry) => entry.type === "branch_summary");
		const rawTailTokens = estimateRawTailTokens(branchEntries);
		const observationTokens = estimateObservationTokens(lastCompaction?.summary);

		const compactionDetails =
			lastCompaction && isObservationalCompactionDetails(lastCompaction.details)
				? {
					strategy: lastCompaction.details.strategy,
					model: lastCompaction.details.model,
					observationCount: lastCompaction.details.observationCount,
					reflectorRan: lastCompaction.details.reflectorRan,
					reflectionMode: lastCompaction.details.reflectionMode,
					observationsDropped: lastCompaction.details.observationsDropped,
					isSplitTurn: lastCompaction.details.isSplitTurn,
					usedPreviousSummary: lastCompaction.details.usedPreviousSummary,
					generatedAt: lastCompaction.details.generatedAt,
				}
				: undefined;

		const branchSummaryDetails =
			lastBranchSummary && isObservationalBranchDetails(lastBranchSummary.details)
				? {
					strategy: lastBranchSummary.details.strategy,
					model: lastBranchSummary.details.model,
					observationCount: lastBranchSummary.details.observationCount,
					entryCount: lastBranchSummary.details.entryCount,
					generatedAt: lastBranchSummary.details.generatedAt,
				}
				: undefined;

		return {
			autoObserverEnabled,
			observerTriggerTokens,
			rawTailTokens,
			reflectorTriggerTokens,
			observationTokens,
			autoCompactInFlight,
			forceReflectPending: forceReflectNextCompaction,
			lastCompaction: lastCompaction
				? {
					id: lastCompaction.id,
					timestamp: lastCompaction.timestamp,
					tokensBefore: lastCompaction.tokensBefore,
					fromExtension: lastCompaction.fromHook,
					details: compactionDetails,
				}
				: undefined,
			lastBranchSummary: lastBranchSummary
				? {
					id: lastBranchSummary.id,
					timestamp: lastBranchSummary.timestamp,
					details: branchSummaryDetails,
				}
				: undefined,
			observations: lastCompaction?.summary ? stripFileTags(lastCompaction.summary) : undefined,
		};
	};

	const showStatusOverlay = async (ctx: ExtensionContext): Promise<void> => {
		if (!ctx.hasUI) return;
		if (statusOverlayOpen) return;

		const snapshot = buildStatusSnapshot(ctx);
		statusOverlayOpen = true;
		try {
			await ctx.ui.custom<null>(
				(_tui, _theme, _keys, done) => new ObservationMemoryOverlay(snapshot, done),
				{ overlay: true },
			);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Unable to render obs overlay: ${message}`, "error");
		} finally {
			statusOverlayOpen = false;
		}
	};

	pi.on("session_start", async (_event, ctx) => {
		const enabledFlag = pi.getFlag("obs-auto-compact");
		if (typeof enabledFlag === "boolean") {
			autoObserverEnabled = enabledFlag;
		}

		const modeFlag = pi.getFlag("obs-mode");
		if (typeof modeFlag === "string") {
			const parsedMode = parseAutoCompactionMode(modeFlag);
			if (parsedMode) {
				autoCompactionMode = parsedMode;
			} else if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory: invalid --obs-mode value "${modeFlag}". Keeping ${autoCompactionMode}.`,
					"warning",
				);
			}
		}

		const observerFlag = pi.getFlag("obs-observer-threshold");
		if (typeof observerFlag === "string") {
			const parsed = parseTokenCount(observerFlag);
			if (parsed !== undefined) {
				observerTriggerTokens = parsed;
			} else if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory: invalid --obs-observer-threshold value "${observerFlag}". Keeping ${formatTokenCount(observerTriggerTokens)}.`,
					"warning",
				);
			}
		}

		const reflectorFlag = pi.getFlag("obs-reflector-threshold");
		if (typeof reflectorFlag === "string") {
			const parsed = parseTokenCount(reflectorFlag);
			if (parsed !== undefined) {
				reflectorTriggerTokens = parsed;
			} else if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory: invalid --obs-reflector-threshold value "${reflectorFlag}". Keeping ${formatTokenCount(reflectorTriggerTokens)}.`,
					"warning",
				);
			}
		}

		const retainFlag = pi.getFlag("obs-retain-raw-tail");
		if (typeof retainFlag === "string") {
			const parsed = parseRetainRawTailTokenCount(retainFlag);
			if (parsed !== undefined) {
				rawTailRetainTokens = parsed;
			} else if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory: invalid --obs-retain-raw-tail value "${retainFlag}". Keeping ${formatTokenCount(rawTailRetainTokens)}.`,
					"warning",
				);
			}
		}
	});

	pi.on("agent_end", async (_event, ctx) => {
		if (autoCompactionMode !== "buffered") return;
		setTimeout(() => {
			if (!ctx.isIdle()) return;
			triggerObserverAutoCompaction(ctx);
		}, 0);
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const { preparation, customInstructions, signal } = event;
		const { messagesToSummarize, turnPrefixMessages, previousSummary, settings } = preparation;

		if (!ctx.model) {
			if (ctx.hasUI) {
				ctx.ui.notify("Observational memory: no active model, falling back to default compaction", "warning");
			}
			return;
		}

		const apiKey = await ctx.modelRegistry.getApiKey(ctx.model);
		if (!apiKey) {
			if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory: missing API key for ${buildModelRef(ctx.model)}, falling back to default compaction`,
					"warning",
				);
			}
			return;
		}

		const allMessages = [...messagesToSummarize, ...turnPrefixMessages];
		if (allMessages.length === 0 && !previousSummary) {
			return;
		}

		const conversationText = serializeConversation(convertToLlm(allMessages));
		const previousSummaryForPrompt = previousSummary ? stripFileTags(previousSummary) : undefined;
		const previousObservationTokens = estimateObservationTokens(previousSummaryForPrompt);
		const promptText = buildCompactionPrompt(conversationText, {
			previousSummary: previousSummaryForPrompt,
			customInstructions,
			isSplitTurn: preparation.isSplitTurn,
			forceReflect: forceReflectNextCompaction,
		});

		const maxTokens = Math.max(512, Math.floor(settings.reserveTokens * 0.8));

		try {
			const rawSummary = await summarizeWithModel(ctx.model, apiKey, promptText, maxTokens, signal);
			const normalized = normalizeSummary(rawSummary);

			const candidateObservationTokens = estimateObservationTokens(normalized);
			const shouldReflectByThreshold =
				Math.max(previousObservationTokens, candidateObservationTokens) >= reflectorTriggerTokens;
			const reflectionMode: ReflectionMode = forceReflectNextCompaction
				? "forced"
				: shouldReflectByThreshold
					? "threshold"
					: "none";

			const reflected = reflectSummary(normalized, reflectionMode);
			const summaryCore = reflected.summary;
			const summary = summaryCore + formatFileOperations(preparation.fileOps, previousSummary);

			const details: ObservationalCompactionDetails = {
				schemaVersion: DETAILS_SCHEMA_VERSION,
				strategy: "observational-memory",
				model: buildModelRef(ctx.model),
				observationCount: reflected.after,
				observationCountBefore: reflected.before,
				observationCountAfter: reflected.after,
				observationsDropped: reflected.dropped,
				reflectorRan: reflectionMode !== "none",
				reflectionMode,
				generatedAt: new Date().toISOString(),
				isSplitTurn: preparation.isSplitTurn,
				usedPreviousSummary: Boolean(previousSummary),
			};

			const compaction: CompactionResult<ObservationalCompactionDetails> = {
				summary,
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				details,
			};

			forceReflectNextCompaction = false;
			return { compaction };
		} catch (error) {
			if (!signal.aborted && ctx.hasUI) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.ui.notify(`Observational memory failed: ${message}. Using default compaction.`, "error");
			}
			return;
		}
	});

	pi.on("session_before_tree", async (event, ctx) => {
		const { preparation, signal } = event;
		if (!preparation.userWantsSummary) return;
		if (preparation.entriesToSummarize.length === 0) return;

		if (!ctx.model) {
			if (ctx.hasUI) {
				ctx.ui.notify("Observational memory(tree): no active model, using default tree summary", "warning");
			}
			return;
		}

		const apiKey = await ctx.modelRegistry.getApiKey(ctx.model);
		if (!apiKey) {
			if (ctx.hasUI) {
				ctx.ui.notify(
					`Observational memory(tree): missing API key for ${buildModelRef(ctx.model)}, using default tree summary`,
					"warning",
				);
			}
			return;
		}

		const tokenBudget =
			ctx.model.contextWindow > DEFAULT_RESERVE_TOKENS ? ctx.model.contextWindow - DEFAULT_RESERVE_TOKENS : 0;
		const branchPreparation = prepareBranchEntries(preparation.entriesToSummarize, tokenBudget);
		if (branchPreparation.messages.length === 0) return;

		const conversationText = serializeConversation(convertToLlm(branchPreparation.messages));
		const promptText = buildTreePrompt(conversationText, {
			customInstructions: preparation.customInstructions,
			replaceInstructions: preparation.replaceInstructions,
		});

		const maxTokens = Math.max(512, Math.floor(DEFAULT_RESERVE_TOKENS * 0.6));

		try {
			const rawSummary = await summarizeWithModel(ctx.model, apiKey, promptText, maxTokens, signal);
			const summaryCore = normalizeSummary(rawSummary);
			const summary = summaryCore + formatFileOperations(branchPreparation.fileOps);
			const details: ObservationalBranchSummaryDetails = {
				schemaVersion: DETAILS_SCHEMA_VERSION,
				strategy: "observational-memory-tree",
				model: buildModelRef(ctx.model),
				observationCount: countObservationLines(summaryCore),
				generatedAt: new Date().toISOString(),
				entryCount: preparation.entriesToSummarize.length,
			};

			return { summary: { summary, details } };
		} catch (error) {
			if (!signal.aborted && ctx.hasUI) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.ui.notify(`Observational memory(tree) failed: ${message}. Using default tree summary.`, "error");
			}
			return;
		}
	});

	pi.on("session_compact", async () => {
		forceReflectNextCompaction = false;
		autoCompactInFlight = false;
	});

	pi.registerCommand(OBS_STATUS_COMMAND, {
		description: "Show observational-memory compaction and tree summary status",
		handler: async (_args, ctx) => {
			if (ctx.hasUI) {
				await showStatusOverlay(ctx);
				return;
			}

			const branchEntries = ctx.sessionManager.getBranch();
			const lastCompaction = [...branchEntries].reverse().find((entry) => entry.type === "compaction");
			const lastBranchSummary = [...branchEntries].reverse().find((entry) => entry.type === "branch_summary");
			const rawTailTokens = estimateRawTailTokens(branchEntries);
			const observationTokens = estimateObservationTokens(lastCompaction?.summary);
			const activationThreshold = getObserverActivationThreshold(observerTriggerTokens, rawTailRetainTokens);

			const lines = [
				"Observational Memory Status",
				"",
				`Observer auto-trigger: ${autoObserverEnabled ? "on" : "off"}`,
				`Observer mode: ${autoCompactionMode}`,
				`Observer threshold: ${formatTokenCount(observerTriggerTokens)}`,
				`Raw-tail retain: ${formatTokenCount(rawTailRetainTokens)}`,
				`Observer activation threshold: ${formatTokenCount(activationThreshold)}`,
				`Raw tail now: ${formatTokenCount(rawTailTokens)}`,
				`Reflector threshold: ${formatTokenCount(reflectorTriggerTokens)}`,
				`Observation block now: ${formatTokenCount(observationTokens)}`,
				`Auto-compact in flight: ${autoCompactInFlight ? "yes" : "no"}`,
				`Force-reflect pending: ${forceReflectNextCompaction ? "yes" : "no"}`,
			];

			if (lastCompaction) {
				lines.push(
					"",
					"Last compaction:",
					`  id: ${lastCompaction.id}`,
					`  timestamp: ${lastCompaction.timestamp}`,
					`  tokensBefore: ${lastCompaction.tokensBefore.toLocaleString()}`,
					`  fromExtension: ${lastCompaction.fromHook ? "yes" : "no"}`,
				);

				const details = isObservationalCompactionDetails(lastCompaction.details) ? lastCompaction.details : undefined;
				if (details) {
					lines.push(
						`  strategy: ${details.strategy}`,
						`  model: ${details.model}`,
						`  observations: ${details.observationCount}`,
						`  reflectorRan: ${details.reflectorRan ? "yes" : "no"} (${details.reflectionMode})`,
						`  dropped: ${details.observationsDropped}`,
						`  splitTurn: ${details.isSplitTurn ? "yes" : "no"}`,
						`  usedPreviousSummary: ${details.usedPreviousSummary ? "yes" : "no"}`,
						`  generatedAt: ${details.generatedAt}`,
					);
				}
			} else {
				lines.push("", "No compaction entries found in current branch.");
			}

			if (lastBranchSummary) {
				const details = isObservationalBranchDetails(lastBranchSummary.details) ? lastBranchSummary.details : undefined;
				lines.push(
					"",
					"Last branch summary:",
					`  id: ${lastBranchSummary.id}`,
					`  timestamp: ${lastBranchSummary.timestamp}`,
				);
				if (details) {
					lines.push(
						`  strategy: ${details.strategy}`,
						`  model: ${details.model}`,
						`  observations: ${details.observationCount}`,
						`  entryCount: ${details.entryCount}`,
						`  generatedAt: ${details.generatedAt}`,
					);
				}
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	pi.registerShortcut(OBS_STATUS_SHORTCUT, {
		description: "Open observational-memory status overlay",
		handler: async (ctx) => {
			await showStatusOverlay(ctx);
		},
	});

	pi.registerCommand(OBS_AUTO_COMPACT_COMMAND, {
		description: "Show or set observer/reflector thresholds, mode, and raw-tail retention",
		handler: async (args, ctx) => {
			const raw = args.trim();
			if (raw.length === 0) {
				const activationThreshold = getObserverActivationThreshold(observerTriggerTokens, rawTailRetainTokens);
				ctx.ui.notify(
					[
						"Observational auto-compaction",
						`observer trigger enabled: ${autoObserverEnabled ? "yes" : "no"}`,
						`mode: ${autoCompactionMode}`,
						`observer threshold: ${formatTokenCount(observerTriggerTokens)}`,
						`reflector threshold: ${formatTokenCount(reflectorTriggerTokens)}`,
						`raw-tail retain: ${formatTokenCount(rawTailRetainTokens)}`,
						`observer activation threshold: ${formatTokenCount(activationThreshold)}`,
						`allowed threshold range: ${formatTokenCount(AUTO_TOKENS_MIN)} - ${formatTokenCount(AUTO_TOKENS_MAX)}`,
						"usage: /obs-auto-compact [on|off] [mode] [observerTokens] [reflectorTokens] [retainTokens]",
						"keyed usage: /obs-auto-compact mode=buffered observer=30k reflector=40k retain=8k",
						"examples: /obs-auto-compact on buffered 30k 40k 8k | /obs-auto-compact retain=0",
					].join("\n"),
					"info",
				);
				return;
			}

			let nextEnabled = autoObserverEnabled;
			let nextMode = autoCompactionMode;
			let nextObserver = observerTriggerTokens;
			let nextReflector = reflectorTriggerTokens;
			let nextRetain = rawTailRetainTokens;
			let positionalTokenCount = 0;

			for (const token of raw.split(/\s+/)) {
				const enabled = parseEnabledToken(token);
				if (enabled !== undefined) {
					nextEnabled = enabled;
					continue;
				}

				const mode = parseAutoCompactionMode(token);
				if (mode) {
					nextMode = mode;
					continue;
				}

				const equalIndex = token.indexOf("=");
				if (equalIndex > 0) {
					const key = token.slice(0, equalIndex).trim().toLowerCase();
					const value = token.slice(equalIndex + 1).trim();

					if (key === "mode" || key === "strategy") {
						const parsedMode = parseAutoCompactionMode(value);
						if (!parsedMode) {
							ctx.ui.notify(`Invalid mode "${value}". Use buffered or blocking.`, "warning");
							return;
						}
						nextMode = parsedMode;
						continue;
					}

					if (key === "enabled" || key === "auto") {
						const parsedEnabled = parseEnabledToken(value);
						if (parsedEnabled === undefined) {
							ctx.ui.notify(`Invalid enabled value "${value}". Use on/off.`, "warning");
							return;
						}
						nextEnabled = parsedEnabled;
						continue;
					}

					if (key === "observer" || key === "obs" || key === "raw") {
						const parsed = parseTokenCount(value);
						if (parsed === undefined) {
							ctx.ui.notify(`Invalid observer threshold "${value}". Use values like 30000 or 30k.`, "warning");
							return;
						}
						nextObserver = parsed;
						continue;
					}

					if (key === "reflector" || key === "reflect" || key === "observations") {
						const parsed = parseTokenCount(value);
						if (parsed === undefined) {
							ctx.ui.notify(`Invalid reflector threshold "${value}". Use values like 40000 or 40k.`, "warning");
							return;
						}
						nextReflector = parsed;
						continue;
					}

					if (key === "retain" || key === "keep" || key === "buffer" || key === "partial") {
						const parsed = parseRetainRawTailTokenCount(value);
						if (parsed === undefined) {
							ctx.ui.notify(`Invalid raw-tail retain value "${value}". Use values like 8000, 8k, or 0.`, "warning");
							return;
						}
						nextRetain = parsed;
						continue;
					}

					ctx.ui.notify(
						`Unknown keyed argument "${key}". Use mode=..., observer=..., reflector=..., retain=..., enabled=...`,
						"warning",
					);
					return;
				}

				if (positionalTokenCount === 0) {
					const parsed = parseTokenCount(token);
					if (parsed !== undefined) {
						nextObserver = parsed;
						positionalTokenCount++;
						continue;
					}
				}
				if (positionalTokenCount === 1) {
					const parsed = parseTokenCount(token);
					if (parsed !== undefined) {
						nextReflector = parsed;
						positionalTokenCount++;
						continue;
					}
				}
				if (positionalTokenCount === 2) {
					const parsed = parseRetainRawTailTokenCount(token);
					if (parsed !== undefined) {
						nextRetain = parsed;
						positionalTokenCount++;
						continue;
					}
				}

				ctx.ui.notify(
					`Invalid argument "${token}". Use on/off, mode, token counts (30000/30k), or keyed forms: observer=/reflector=/retain=/mode=.`,
					"warning",
				);
				return;
			}

			autoObserverEnabled = nextEnabled;
			autoCompactionMode = nextMode;
			observerTriggerTokens = nextObserver;
			reflectorTriggerTokens = nextReflector;
			rawTailRetainTokens = nextRetain;
			if (!autoObserverEnabled) {
				autoCompactInFlight = false;
			}

			const activationThreshold = getObserverActivationThreshold(observerTriggerTokens, rawTailRetainTokens);
			ctx.ui.notify(
				[
					"Observational auto-compaction updated:",
					`- observer trigger: ${autoObserverEnabled ? "on" : "off"}`,
					`- mode: ${autoCompactionMode}`,
					`- observer threshold: ${formatTokenCount(observerTriggerTokens)}`,
					`- reflector threshold: ${formatTokenCount(reflectorTriggerTokens)}`,
					`- raw-tail retain: ${formatTokenCount(rawTailRetainTokens)}`,
					`- observer activation threshold: ${formatTokenCount(activationThreshold)}`,
				].join("\n"),
				"info",
			);

			if (autoObserverEnabled && autoCompactionMode === "buffered") {
				setTimeout(() => {
					if (!ctx.isIdle()) return;
					triggerObserverAutoCompaction(ctx);
				}, 0);
			}
		},
	});

	pi.registerCommand(OBS_MODE_COMMAND, {
		description: "Show or set observer auto-compaction mode (buffered|blocking)",
		handler: async (args, ctx) => {
			const raw = args.trim();
			if (raw.length === 0) {
				ctx.ui.notify(
					[
						"Observational mode",
						`current: ${autoCompactionMode}`,
						"buffered: observer runs in background on agent_end",
						"blocking: disable observer background trigger; only regular/manual compaction runs",
						"usage: /obs-mode buffered|blocking",
					].join("\n"),
					"info",
				);
				return;
			}

			const parsedMode = parseAutoCompactionMode(raw);
			if (!parsedMode) {
				ctx.ui.notify(`Invalid mode "${raw}". Use buffered or blocking.`, "warning");
				return;
			}

			autoCompactionMode = parsedMode;
			if (autoCompactionMode === "blocking") {
				autoCompactInFlight = false;
			}

			ctx.ui.notify(`Observational mode updated: ${autoCompactionMode}.`, "info");
		},
	});

	pi.registerCommand(OBS_VIEW_COMMAND, {
		description: "Show latest observation summary from compaction",
		handler: async (args, ctx) => {
			const branchEntries = ctx.sessionManager.getBranch();
			const lastCompaction = [...branchEntries].reverse().find((entry) => entry.type === "compaction");
			if (!lastCompaction) {
				ctx.ui.notify("No compaction found in current branch.", "warning");
				return;
			}

			let includeFileTags = false;
			let section: "all" | "observations" = "all";
			let maxLines = 160;
			for (const token of args
				.trim()
				.split(/\s+/)
				.filter((part) => part.length > 0)) {
				const normalized = token.toLowerCase();
				if (["raw", "full", "tags"].includes(normalized)) {
					includeFileTags = true;
					continue;
				}
				if (["obs", "observations"].includes(normalized)) {
					section = "observations";
					continue;
				}

				const parsedNumber = Number.parseInt(normalized, 10);
				if (Number.isFinite(parsedNumber) && parsedNumber > 0) {
					maxLines = Math.max(10, Math.min(400, parsedNumber));
					continue;
				}

				ctx.ui.notify(`Unknown argument "${token}". Use: raw|obs|<maxLines>. Example: /obs-view obs 120`, "warning");
				return;
			}

			const baseSummary = includeFileTags ? lastCompaction.summary : stripFileTags(lastCompaction.summary);
			const rendered =
				section === "observations"
					? [
							"## Observations",
							extractSection(baseSummary, "## Observations", "## Open Threads") ||
								"Date: unknown\n- 🟡 No observations found.",
						].join("\n")
					: baseSummary;

			const lines = rendered.split("\n");
			const clipped = lines.slice(0, maxLines);
			const truncatedNote =
				lines.length > clipped.length
					? `\n\n... truncated ${lines.length - clipped.length} lines. Pass a higher limit (e.g. /obs-view ${lines.length}).`
					: "";

			ctx.ui.notify(
				[
					`Observational view (${section}, ${includeFileTags ? "with" : "without"} file tags)`,
					`Compaction id: ${lastCompaction.id}`,
					"",
					clipped.join("\n"),
					truncatedNote,
				].join("\n"),
				"info",
			);
		},
	});

	pi.registerCommand(OBS_REFLECT_COMMAND, {
		description: "Force aggressive observation reflection on next compaction and trigger compaction",
		handler: async (args, ctx) => {
			forceReflectNextCompaction = true;
			const extra = args.trim();
			const customInstructions = [
				"Aggressive reflector mode: deduplicate observations, remove stale low-priority details, preserve critical constraints and blockers.",
				extra.length > 0 ? `Extra focus: ${extra}` : undefined,
			]
				.filter((line): line is string => Boolean(line))
				.join("\n");

			if (ctx.hasUI) {
				ctx.ui.notify("Queued forced reflection and triggering compaction...", "info");
			}

			ctx.compact({
				customInstructions,
				onComplete: (result) => {
					if (ctx.hasUI) {
						ctx.ui.notify(
							`Forced reflection compaction complete (${result.tokensBefore.toLocaleString()} tokens before).`,
							"info",
						);
					}
				},
				onError: (error) => {
					if (ctx.hasUI) {
						ctx.ui.notify(`Forced reflection compaction failed: ${error.message}`, "error");
					}
				},
			});
		},
	});
}

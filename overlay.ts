import { Key, matchesKey } from "@mariozechner/pi-tui";

const ANSI_RE = /\x1b\[[0-9;]*m/g;

function color(code: string, text: string): string {
	return `\x1b[${code}m${text}\x1b[0m`;
}

function bold(text: string): string {
	return `\x1b[1m${text}\x1b[22m`;
}

function dim(text: string): string {
	return color("2", text);
}

function visibleLength(text: string): number {
	return text.replace(ANSI_RE, "").length;
}

function clipAnsi(text: string, width: number): string {
	if (visibleLength(text) <= width) return text;
	let visible = 0;
	let i = 0;
	let out = "";
	while (i < text.length && visible < width) {
		const ch = text[i];
		if (ch === "\u001b" && text[i + 1] === "[") {
			let j = i + 2;
			while (j < text.length && text[j] !== "m") j++;
			if (j < text.length) {
				out += text.slice(i, j + 1);
				i = j + 1;
				continue;
			}
		}
		out += ch;
		visible += 1;
		i += 1;
	}
	return `${out}\x1b[0m`;
}

function padRight(text: string, width: number): string {
	const clipped = clipAnsi(text, width);
	const pad = Math.max(0, width - visibleLength(clipped));
	return clipped + " ".repeat(pad);
}

function wrapPlain(text: string, width: number): string[] {
	if (!text) return [""];
	if (text.length <= width) return [text];

	const words = text.split(/\s+/).filter((word) => word.length > 0);
	if (words.length === 0) return [""];

	const lines: string[] = [];
	let current = "";

	for (const word of words) {
		if (!current) {
			if (word.length <= width) {
				current = word;
				continue;
			}
			for (let i = 0; i < word.length; i += width) {
				lines.push(word.slice(i, i + width));
			}
			continue;
		}

		const candidate = `${current} ${word}`;
		if (candidate.length <= width) {
			current = candidate;
			continue;
		}

		lines.push(current);
		if (word.length <= width) {
			current = word;
		} else {
			for (let i = 0; i < word.length; i += width) {
				lines.push(word.slice(i, i + width));
			}
			current = "";
		}
	}

	if (current) lines.push(current);
	return lines;
}

function formatTokenCount(tokens: number): string {
	return `${tokens.toLocaleString()} tokens`;
}

function meter(current: number, total: number, width = 26): string {
	const ratio = total > 0 ? Math.min(current / total, 1) : 0;
	const filled = Math.max(0, Math.min(width, Math.round(ratio * width)));
	const colorCode = ratio >= 0.95 ? "31" : ratio >= 0.7 ? "33" : "32";
	return `${color("2", "[")}${color(colorCode, "█".repeat(filled))}${color("2", "░".repeat(width - filled))}${color("2", "]")} ${color(colorCode, `${Math.round(ratio * 100)}%`)}`;
}

type Tab = "status" | "observations";

type Severity = "normal" | "heading" | "red" | "yellow" | "green" | "muted";

interface StyledLine {
	text: string;
	severity?: Severity;
}

interface CompactionOverlayDetails {
	strategy?: string;
	model?: string;
	observationCount?: number;
	reflectorRan?: boolean;
	reflectionMode?: string;
	observationsDropped?: number;
	isSplitTurn?: boolean;
	usedPreviousSummary?: boolean;
	generatedAt?: string;
}

interface BranchOverlayDetails {
	strategy?: string;
	model?: string;
	observationCount?: number;
	entryCount?: number;
	generatedAt?: string;
}

export interface ObservationMemoryOverlaySnapshot {
	autoObserverEnabled: boolean;
	observerTriggerTokens: number;
	rawTailTokens: number;
	reflectorTriggerTokens: number;
	observationTokens: number;
	autoCompactInFlight: boolean;
	forceReflectPending: boolean;
	lastCompaction?: {
		id: string;
		timestamp: number | string;
		tokensBefore: number;
		fromExtension: boolean;
		details?: CompactionOverlayDetails;
	};
	lastBranchSummary?: {
		id: string;
		timestamp: number | string;
		details?: BranchOverlayDetails;
	};
	observations?: string;
}

function styleLine(line: StyledLine): string {
	switch (line.severity) {
		case "heading":
			return bold(color("36", line.text));
		case "red":
			return color("31", line.text);
		case "yellow":
			return color("33", line.text);
		case "green":
			return color("32", line.text);
		case "muted":
			return dim(line.text);
		default:
			return line.text;
	}
}

function buildStatusLines(snapshot: ObservationMemoryOverlaySnapshot): StyledLine[] {
	const lines: StyledLine[] = [
		{ text: "Observer/Reflector" },
		{ text: `Observer trigger: ${snapshot.autoObserverEnabled ? "on" : "off"}` },
		{ text: `Observer threshold: ${formatTokenCount(snapshot.observerTriggerTokens)}` },
		{ text: `Raw tail now: ${formatTokenCount(snapshot.rawTailTokens)}` },
		{ text: meter(snapshot.rawTailTokens, snapshot.observerTriggerTokens), severity: "normal" },
		{ text: "" },
		{ text: `Reflector threshold: ${formatTokenCount(snapshot.reflectorTriggerTokens)}` },
		{ text: `Observation block: ${formatTokenCount(snapshot.observationTokens)}` },
		{ text: meter(snapshot.observationTokens, snapshot.reflectorTriggerTokens), severity: "normal" },
		{ text: "" },
		{ text: `Auto-compact in flight: ${snapshot.autoCompactInFlight ? "yes" : "no"}` },
		{ text: `Force-reflect pending: ${snapshot.forceReflectPending ? "yes" : "no"}` },
		{ text: "" },
	];

	if (snapshot.lastCompaction) {
		lines.push(
			{ text: "Last compaction", severity: "heading" },
			{ text: `id: ${snapshot.lastCompaction.id}` },
			{ text: `timestamp: ${new Date(snapshot.lastCompaction.timestamp).toLocaleString()}` },
			{ text: `tokensBefore: ${snapshot.lastCompaction.tokensBefore.toLocaleString()}` },
			{ text: `fromExtension: ${snapshot.lastCompaction.fromExtension ? "yes" : "no"}` },
		);

		if (snapshot.lastCompaction.details) {
			const details = snapshot.lastCompaction.details;
			lines.push(
				{ text: `strategy: ${details.strategy ?? "unknown"}`, severity: "muted" },
				{ text: `model: ${details.model ?? "unknown"}`, severity: "muted" },
				{ text: `observations: ${details.observationCount ?? 0}`, severity: "muted" },
				{
					text: `reflector: ${details.reflectorRan ? "yes" : "no"}${details.reflectionMode ? ` (${details.reflectionMode})` : ""}`,
					severity: "muted",
				},
				{ text: `dropped: ${details.observationsDropped ?? 0}`, severity: "muted" },
				{ text: `splitTurn: ${details.isSplitTurn ? "yes" : "no"}`, severity: "muted" },
				{ text: `usedPreviousSummary: ${details.usedPreviousSummary ? "yes" : "no"}`, severity: "muted" },
			);
			if (details.generatedAt) {
				lines.push({ text: `generatedAt: ${details.generatedAt}`, severity: "muted" });
			}
		}

		lines.push({ text: "" });
	} else {
		lines.push({ text: "No compaction entries found in current branch.", severity: "yellow" }, { text: "" });
	}

	if (snapshot.lastBranchSummary) {
		lines.push(
			{ text: "Last branch summary", severity: "heading" },
			{ text: `id: ${snapshot.lastBranchSummary.id}` },
			{ text: `timestamp: ${new Date(snapshot.lastBranchSummary.timestamp).toLocaleString()}` },
		);
		if (snapshot.lastBranchSummary.details) {
			const details = snapshot.lastBranchSummary.details;
			lines.push(
				{ text: `strategy: ${details.strategy ?? "unknown"}`, severity: "muted" },
				{ text: `model: ${details.model ?? "unknown"}`, severity: "muted" },
				{ text: `observations: ${details.observationCount ?? 0}`, severity: "muted" },
				{ text: `entryCount: ${details.entryCount ?? 0}`, severity: "muted" },
			);
			if (details.generatedAt) {
				lines.push({ text: `generatedAt: ${details.generatedAt}`, severity: "muted" });
			}
		}
	}

	return lines;
}

function buildObservationLines(summary: string | undefined): StyledLine[] {
	if (!summary || summary.trim().length === 0) {
		return [{ text: "No observations in the latest compaction yet.", severity: "yellow" }];
	}

	return summary.split("\n").map((line) => {
		if (line.startsWith("## ")) return { text: line, severity: "heading" as const };
		if (line.startsWith("Date:")) return { text: line, severity: "muted" as const };
		if (line.startsWith("- 🔴")) return { text: line, severity: "red" as const };
		if (line.startsWith("- 🟡")) return { text: line, severity: "yellow" as const };
		if (line.startsWith("- 🟢")) return { text: line, severity: "green" as const };
		if (/^\d+\.\s+/.test(line)) return { text: line, severity: "green" as const };
		if (line.trim().length === 0) return { text: "" };
		return { text: line };
	});
}

function wrapStyledLines(lines: StyledLine[], width: number): string[] {
	const wrapped: string[] = [];
	for (const line of lines) {
		const parts = wrapPlain(line.text, width);
		for (const part of parts) {
			wrapped.push(styleLine({ text: part, severity: line.severity }));
		}
	}
	return wrapped;
}

export class ObservationMemoryOverlay {
	private readonly maxWidth = 98;
	private readonly contentRows = 20;
	private tab: Tab = "status";
	private scrollOffset = 0;
	private cacheWidth = 0;
	private statusLines: string[] = [];
	private observationLines: string[] = [];

	constructor(
		private snapshot: ObservationMemoryOverlaySnapshot,
		private done: (result: null) => void,
	) {}

	handleInput(data: string): void {
		if (matchesKey(data, Key.escape) || data === "q") {
			this.done(null);
			return;
		}

		if (matchesKey(data, Key.tab) || data === "1" || data === "2") {
			this.tab = data === "1" ? "status" : data === "2" ? "observations" : this.tab === "status" ? "observations" : "status";
			this.scrollOffset = 0;
			return;
		}

		if (matchesKey(data, Key.up) || data === "k") {
			this.scrollOffset = Math.max(0, this.scrollOffset - 1);
			return;
		}
		if (matchesKey(data, Key.down) || data === "j") {
			this.scrollOffset += 1;
			return;
		}
		if (matchesKey(data, "pageup") || matchesKey(data, Key.ctrl("u"))) {
			this.scrollOffset = Math.max(0, this.scrollOffset - 8);
			return;
		}
		if (matchesKey(data, "pagedown") || matchesKey(data, Key.ctrl("d"))) {
			this.scrollOffset += 8;
			return;
		}
		if (data === "g") {
			this.scrollOffset = 0;
			return;
		}
		if (data === "G") {
			const maxScroll = Math.max(0, this.activeLines().length - this.contentRows);
			this.scrollOffset = maxScroll;
		}
	}

	render(width: number): string[] {
		if (width < 20) {
			return [padRight("Obs Memory", width)];
		}

		const frameWidth = Math.min(this.maxWidth, width);
		const innerWidth = frameWidth - 2;
		const contentWidth = Math.max(1, innerWidth - 2);

		if (this.cacheWidth !== contentWidth) {
			this.cacheWidth = contentWidth;
			this.statusLines = wrapStyledLines(buildStatusLines(this.snapshot), contentWidth);
			this.observationLines = wrapStyledLines(buildObservationLines(this.snapshot.observations), contentWidth);
		}

		const lines = this.activeLines();
		const maxScroll = Math.max(0, lines.length - this.contentRows);
		if (this.scrollOffset > maxScroll) this.scrollOffset = maxScroll;

		const visible = lines.slice(this.scrollOffset, this.scrollOffset + this.contentRows);
		const statusTab = this.tab === "status" ? bold(color("36", "● Status")) : dim("○ Status");
		const obsTab = this.tab === "observations" ? bold(color("36", "● Observations")) : dim("○ Observations");

		const out: string[] = [];
		const title = bold(color("36", " 🧠 Observational Memory "));
		const sidePad = Math.max(0, innerWidth - visibleLength(title));
		const leftPad = Math.floor(sidePad / 2);
		const rightPad = sidePad - leftPad;

		out.push(dim("╭") + dim("─".repeat(leftPad)) + title + dim("─".repeat(rightPad)) + dim("╮"));
		out.push(dim("│") + " " + padRight(`${statusTab}  ${obsTab}`, innerWidth - 1) + dim("│"));
		out.push(dim("├") + dim("─".repeat(innerWidth)) + dim("┤"));

		for (let i = 0; i < this.contentRows; i++) {
			const line = visible[i] ?? "";
			out.push(dim("│") + " " + padRight(line, innerWidth - 1) + dim("│"));
		}

		const rangeStart = lines.length === 0 ? 0 : this.scrollOffset + 1;
		const rangeEnd = Math.min(lines.length, this.scrollOffset + this.contentRows);
		const footer = dim(` ${rangeStart}-${rangeEnd} / ${lines.length} `);
		const footerPad = Math.max(0, innerWidth - visibleLength(footer));
		out.push(dim("├") + dim("─".repeat(Math.floor(footerPad / 2))) + footer + dim("─".repeat(Math.ceil(footerPad / 2))) + dim("┤"));

		const hints = dim("↑↓/jk scroll  PgUp/PgDn page  tab switch  esc close");
		out.push(dim("│") + " " + padRight(hints, innerWidth - 1) + dim("│"));
		out.push(dim("╰") + dim("─".repeat(innerWidth)) + dim("╯"));
		return out;
	}

	invalidate(): void {
		this.cacheWidth = 0;
		this.statusLines = [];
		this.observationLines = [];
	}

	dispose(): void {}

	private activeLines(): string[] {
		return this.tab === "status" ? this.statusLines : this.observationLines;
	}
}

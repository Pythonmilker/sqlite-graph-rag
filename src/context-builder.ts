import { asSalience, salienceTag, SALIENCE_RANK, type Salience } from './salience';
import type { GraphContext, Hit, Neighbor } from './graph-rag-index';

/**
 * Token-bounded context assembly: the cost and faithfulness core of a retrieval pipeline.
 *
 * Two HARD-SEPARATED blocks from a GraphRAG bundle:
 *   • FACTS     — a structured entity/relationship snapshot from the KG = GROUND TRUTH. Prioritized;
 *                 a name-only fallback per line means a NAME is never dropped, only its description.
 *   • NARRATIVE — prose snippets (session notes, plot beats) = TONE ONLY, never authoritative; the
 *                 first thing trimmed under budget pressure.
 * Splitting them neutralizes the two failure modes of a single context blob: retrieval-miss (facts are
 * regenerated from the KG every turn, not carried in a drifting summary) and summary-drift (narrative is
 * explicitly labelled untrusted-for-fact).
 *
 * Pure + deterministic (no IO) so it unit-tests without a DB and runs IDENTICALLY in the desktop sidecar
 * and the mobile client. The invariant the pivot rests on: the returned block is ALWAYS <= maxTokens,
 * regardless of corpus size — cost scales with the retrieved top-k, not with how much data exists.
 */

const CHARS_PER_TOKEN = 4;
// CJK / Japanese / Korean / fullwidth codepoints tokenize at roughly 1–2.5 REAL tokens EACH — the opposite
// of the ~4-Latin-chars-per-token assumption. A token *ceiling* that under-reports is worse than useless
// (it lets a non-Latin corpus blow the budget while reading "under" it), so we weight these at 2 to
// deliberately OVER-count rather than under-count. Kept in lockstep with the hosted proxy's estInputTokens.
const WIDE_CHAR = /[ᄀ-ᇿ　-〿぀-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]/g;

/** Cheap, tokenizer-free, SCRIPT-AWARE token estimate. Latin ≈ 4 chars/token; CJK/JP/KR ≈ 2 tokens/char
 *  (over-counted on purpose so this can serve as a hard cost ceiling). */
export function estimateTokens(s: string): number {
  if (!s) return 0;
  const wide = (s.match(WIDE_CHAR) ?? []).length;
  return Math.ceil(wide * 2 + (s.length - wide) / CHARS_PER_TOKEN);
}

/** Kinds whose prose is STORY-IN-MOTION (tone), not canonical world state — routed to NARRATIVE. */
const NARRATIVE_KINDS = new Set(['note', 'plot']);

export interface ContextOptions {
  /** Soft ceiling for facts+narrative; narrative (then fact descriptions) trim to fit. Default 1800. */
  budgetTokens?: number;
  /** Hard ceiling the assembled block must never exceed — names get dropped before this is breached. */
  maxTokens?: number;
  /** Max chars of each narrative prose snippet. Default 240. */
  snippetChars?: number;
  /** Max chars of each fact description. Default 160. */
  factChars?: number;
}

export interface ContextResult {
  /** The system-context block to append after the persona prompt. '' when there's nothing grounded. */
  text: string;
  factsTokens: number;
  narrativeTokens: number;
  /** Authoritative measure: estimateTokens(text). Guaranteed <= maxTokens. */
  totalTokens: number;
  factsCount: number;
  narrativeCount: number;
  /** Fact lines kept name-only because their description didn't fit the soft budget. */
  factsTruncated: number;
  /** Narrative snippets omitted to honor the soft budget. */
  narrativeDropped: number;
}

const FACTS_HEADER = '=== WORLD FACTS (authoritative — treat as canon; do not contradict) ===';
const NARRATIVE_HEADER = '=== RECENT NARRATIVE (tone only — NOT authoritative; never cite as fact) ===';

/**
 * THE PROMINENCE RULE — the second half of tiering, and the half that actually changes behaviour.
 *
 * The column alone only reorders lines. Without a stated rule the model reads `· incidental` as decoration
 * and keeps volunteering everything, because "here is a fact about this place" reads as an instruction to
 * use it. So the rule travels attached to the block it governs, emitted only when a tagged line is present:
 * one place, impossible to drift from the data, and free on the ordinary turn where nothing is tagged.
 *
 * The failure this prevents is specific. Every fact retrieved about a location is TRUE, so a model with no
 * emphasis signal recites all of it — the exhaustive room description that reads like an inventory instead
 * of a scene. Salience is what lets a detail be remembered without being performed.
 */
const PROMINENCE_RULE =
  'Tags: `defining` = what the thing IS — lead with it, never contradict it. `incidental` = true but minor — ' +
  'do NOT volunteer it; use it only if asked, or if the players reach for it. Untagged sits between. ' +
  'Listing a fact here is permission to KNOW it, not an instruction to say it.';

const oneLine = (s: string): string => (s ?? '').replace(/\s+/g, ' ').trim();
const clip = (s: string, n: number): string => (s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`);

const EMPTY: ContextResult = {
  text: '',
  factsTokens: 0,
  narrativeTokens: 0,
  totalTokens: 0,
  factsCount: 0,
  narrativeCount: 0,
  factsTruncated: 0,
  narrativeDropped: 0,
};

export function buildContext(ctx: GraphContext | null | undefined, opts: ContextOptions = {}): ContextResult {
  if (!ctx) return { ...EMPTY };
  const budget = Math.max(0, opts.budgetTokens ?? 1800);
  const hardMax = Math.max(budget, opts.maxTokens ?? 2600);
  const snippetChars = opts.snippetChars ?? 240;
  const factChars = opts.factChars ?? 160;

  const seeds: Hit[] = Array.isArray(ctx.seeds) ? ctx.seeds : [];
  const related: Neighbor[] = Array.isArray(ctx.related) ? ctx.related : [];
  if (seeds.length === 0 && related.length === 0) return { ...EMPTY };

  // Source lines. Each fact carries a `nameOnly` fallback so its NAME outlives its description.
  type FactLine = { nameOnly: string; full: string; rank: number };
  const factSource: FactLine[] = [];
  const narrSource: string[] = [];
  let tagged = false;

  /** `[place]` or, when the tier carries meaning, `[place · incidental]`. */
  const bracket = (kind: string, p: Salience | undefined): string => {
    const tag = salienceTag(asSalience(p));
    if (tag) tagged = true;
    return tag ? `[${kind} · ${tag}]` : `[${kind}]`;
  };

  for (const s of seeds) {
    const title = oneLine(s.title) || s.kind;
    const prom = asSalience(s.salience);
    const nameOnly = `- ${title} ${bracket(s.kind, prom)}`;
    const rank = SALIENCE_RANK[prom];
    if (NARRATIVE_KINDS.has(s.kind)) {
      const prose = oneLine(s.snippet);
      if (prose) narrSource.push(`- ${title}: ${clip(prose, snippetChars)}`);
      else factSource.push({ nameOnly, full: nameOnly, rank }); // titled-but-prose-less note → at least anchor the name
    } else {
      const desc = oneLine(s.snippet);
      factSource.push({ nameOnly, full: desc ? `${nameOnly}: ${clip(desc, factChars)}` : nameOnly, rank });
    }
  }
  for (const r of related) {
    const title = oneLine(r.title) || r.kind;
    const rel = oneLine(r.relation);
    const nameOnly = `- ${title} [${r.kind}]`;
    // a graph neighbour arrives via an edge, not a node read, so it carries no salience — rank it as the
    // default rather than inventing one.
    factSource.push({ nameOnly, full: rel ? `${nameOnly} —(${rel})— linked to a retrieved entity` : nameOnly, rank: SALIENCE_RANK.notable });
  }

  // KG MEMORY: order by salience before the budget passes below decide what survives. Retrieval order is
  // preserved WITHIN a tier (a stable sort), so this changes which facts get dropped under pressure — the
  // defining ones outlive the incidental ones — without discarding relevance as the tie-break.
  factSource.sort((a, b) => a.rank - b.rank);

  // ---- Phase 1: every fact NAME (mandatory, bounded only by the hard ceiling). Skip-don't-break so a
  //               long line never starves the shorter names that follow it. ----
  const included: Array<{ src: FactLine; full: boolean }> = [];
  let factsTok = estimateTokens(FACTS_HEADER);
  for (const f of factSource) {
    const t = estimateTokens(f.nameOnly) + 1; // +1 ≈ the line's newline (deliberately over-counts → stays under max)
    if (factsTok + t > hardMax) continue;
    included.push({ src: f, full: false });
    factsTok += t;
  }
  // ---- Phase 2: upgrade names → full descriptions, in priority order, while under the SOFT budget. ----
  for (const inc of included) {
    if (inc.src.full === inc.src.nameOnly) {
      inc.full = true;
      continue;
    }
    const delta = estimateTokens(inc.src.full) - estimateTokens(inc.src.nameOnly);
    if (factsTok + delta > budget) continue;
    inc.full = true;
    factsTok += delta;
  }
  const factText = included.map((inc) => (inc.full ? inc.src.full : inc.src.nameOnly));
  const factsTruncated = included.filter((inc) => !inc.full).length;

  // ---- Phase 2b: the salience rule, charged AFTER the facts it explains. ----
  // Facts come first on purpose. The rule is a fixed ~60-token block, so reserving it up front starved
  // the fact list entirely at a tight ceiling — the block came back EMPTY, which is the one outcome worse
  // than an unexplained tag. So: admit the facts, then take the rule only if a tag actually survived AND
  // it fits under the hard ceiling. Charged here, before narrative, so the ceiling stays exact.
  const ruleEmitted =
    tagged &&
    factText.some((l) => l.includes(' · ')) &&
    factsTok + estimateTokens(PROMINENCE_RULE) + 1 <= hardMax;
  if (ruleEmitted) factsTok += estimateTokens(PROMINENCE_RULE) + 1;

  // ---- Phase 3: narrative snippets fill whatever soft budget remains. ----
  const narrText: string[] = [];
  let narrTok = narrSource.length ? estimateTokens(NARRATIVE_HEADER) : 0;
  let narrativeDropped = 0;
  for (const line of narrSource) {
    const t = estimateTokens(line) + 1;
    if (factsTok + narrTok + t > budget) {
      narrativeDropped++;
      continue;
    }
    narrText.push(line);
    narrTok += t;
  }
  if (narrText.length === 0) narrTok = 0;

  // ---- assemble ----
  const parts: string[] = [];
  if (factText.length) parts.push(`${FACTS_HEADER}\n${ruleEmitted ? `${PROMINENCE_RULE}\n` : ''}${factText.join('\n')}`);
  if (narrText.length) parts.push(`${NARRATIVE_HEADER}\n${narrText.join('\n')}`);
  const text = parts.join('\n\n');

  return {
    text,
    factsTokens: factText.length ? factsTok : 0,
    narrativeTokens: narrTok,
    totalTokens: estimateTokens(text),
    factsCount: factText.length,
    narrativeCount: narrText.length,
    factsTruncated,
    narrativeDropped,
  };
}

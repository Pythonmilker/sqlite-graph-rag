/**
 * SALIENCE — how loudly a node speaks.
 *
 * The problem it solves: storage and recall are not the same question. A fact can be perfectly
 * retrievable and still be wrong to volunteer. Anything relevant enough to answer "what's here?" is
 * relevant enough to be repeated at the reader every single turn, which is what makes a generated
 * assistant tiresome. Ranking alone cannot fix that — so the distinction has to be a property of the
 * DATA, carried through to the model, rather than a scoring accident.
 *
 * WHY IT IS NOT WRITTEN INTO THE INDEXED BODY. Two reasons, both found by watching retrieval run:
 *
 *   1. With no embedder, retrieval is FTS bm25 alone. A label sitting in the body becomes a
 *      SEARCHABLE TOKEN, so a query containing "defining" would bm25-match every defining record on
 *      the strength of its own label.
 *   2. The body is the embedding input. Re-ranking a record would rewrite its text and force a
 *      re-embed. Salience is meant to be adjusted freely; it must stay a cheap UPDATE.
 *
 * It therefore travels as its own column on `nodes`, its own field on a hit, and is rendered once,
 * structurally, into the context block.
 */
export type Salience = 'defining' | 'notable' | 'incidental';

/** The default every unmarked node reads as. Deliberately the middle: an absent value must mean
 *  "behave exactly as before this feature existed". */
export const DEFAULT_SALIENCE: Salience = 'notable';

/** Normalize anything — a database string, a legacy record, undefined — to a valid salience. */
export function asSalience(v: unknown): Salience {
  return v === 'defining' || v === 'incidental' || v === 'notable' ? v : DEFAULT_SALIENCE;
}

/**
 * The compact tag rendered on a context line, e.g. `- The Hollow Well [place · incidental]: …`.
 *
 * `notable` renders empty on purpose: it is the default, and restating it on most lines would spend
 * tokens to say nothing while diluting the two tags that carry meaning.
 */
export function salienceTag(s: Salience): string {
  return s === 'notable' ? '' : s;
}

/** Ordering for budget survival: when the context block has to be trimmed, defining facts outlive
 *  incidental ones. Lower sorts first. */
export const SALIENCE_RANK: Record<Salience, number> = { defining: 0, notable: 1, incidental: 2 };

/** A child fact hanging off a parent record — a remembered note, an annotation, an observation. */
export interface ChildFact {
  id: string;
  text: string;
  salience?: Salience;
}

/** Stable id for a child node, so re-indexing a parent updates its children rather than duplicating. */
export function childNodeId(parentId: string, childId: string): string {
  return `${parentId}#d${childId}`;
}

/**
 * Expand a record's child facts into their own indexable nodes.
 *
 * Children are indexed separately rather than concatenated into the parent's body because a parent
 * with forty accumulated notes would otherwise embed as a single averaged vector that matches
 * nothing well. One fact, one node, one vector.
 */
export function childNodesFor(
  parentId: string,
  parentTitle: string,
  children: readonly ChildFact[] | undefined,
): Array<{ id: string; kind: 'detail'; title: string; body: string; salience: Salience }> {
  if (!Array.isArray(children)) return [];
  const out: Array<{ id: string; kind: 'detail'; title: string; body: string; salience: Salience }> = [];
  for (const c of children) {
    if (!c?.id || !c.text?.trim()) continue;
    out.push({
      id: childNodeId(parentId, c.id),
      kind: 'detail',
      title: parentTitle,
      body: c.text.trim(),
      salience: asSalience(c.salience),
    });
  }
  return out;
}

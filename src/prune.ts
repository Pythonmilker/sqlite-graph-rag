/**
 * WHICH NODES OUTLIVED THE RECORD THEY CAME FROM.
 *
 * Deleting through a handler is not the only way a record disappears. A bulk regeneration drops every old
 * row and mints new ones with new ids, and if it lands through a save path that replaces rather than
 * reconciles, nothing purges the index — so each pass strands another crop. Measured on a real database:
 * 35 orphaned nodes, including three separate copies of the same record from three regenerations, all still
 * in the FTS table and all still matchable. Retrieval then seeds on a record deleted long ago and answers,
 * confidently, from data that no longer exists.
 *
 * The decision is separated from the SQL so it can be reasoned about and tested on its own. Getting it wrong
 * in the unsafe direction empties an entire index, which is worth being able to state as a rule over data
 * rather than as a query you have to read carefully.
 */

import { DETAIL_KIND } from './salience';

/** The minimum a stored node has to tell us. */
export interface NodeRef {
  node_id: string;
  kind: string;
}

/**
 * The node ids to delete: those whose kind we could actually enumerate, and whose record is gone.
 *
 * ⚠️ THE SAFETY RULE IS THE ABSENCE, NOT THE PRESENCE. A kind missing from `liveByKind` was never
 * enumerated — a retired legacy collection, a table this build no longer loads — and its rows must be LEFT
 * ALONE. Treating "I have no live ids for this kind" as "everything of this kind is dead" is how a schema
 * change quietly deletes a corpus's memory. A kind present with an EMPTY set is different and does prune:
 * it means we looked and there is genuinely nothing left.
 *
 * Child (`detail`) nodes are not this function's business — they hang off a parent record and are
 * reconciled with their parent instead (mint ids with `childNodeId`, list what is currently indexed
 * with `childNodeIdsOf`, remove the difference).
 */
export function deadNodes(rows: readonly NodeRef[], liveByKind: ReadonlyMap<string, ReadonlySet<string>>): string[] {
  const dead: string[] = [];
  for (const r of rows) {
    if (r.kind === DETAIL_KIND) continue;
    const live = liveByKind.get(r.kind);
    if (!live) continue; // never enumerated ⇒ never pruned
    if (!live.has(r.node_id)) dead.push(r.node_id);
  }
  return dead;
}

import { describe, it, expect } from 'vitest';
import { deadNodes, type NodeRef } from './prune';

/**
 * Nodes whose source record no longer exists must stop being retrievable.
 *
 * Counted on a real database: 35 orphaned nodes, every one a record name from an earlier regeneration —
 * 'Payments API' appears three times, 'Search Service' twice, 'Auth Service' twice.
 * Every one lives in nodes_fts and is matchable, so retrieval can seed on a record deleted three regenerations ago.
 *
 * The rule is short but the FAILURE MODE IS SEVERE — the unsafe direction empties the whole index — so
 * the safety case gets more tests than the happy path.
 */

const rows = (...specs: [string, string][]): NodeRef[] => specs.map(([node_id, kind]) => ({ node_id, kind }));
const live = (m: Record<string, string[]>) =>
  new Map(Object.entries(m).map(([k, ids]) => [k, new Set(ids)] as const));

describe('what gets pruned', () => {
  it('drops a node whose record is gone and keeps the one that is still there', () => {
    const out = deadNodes(rows(['p1', 'place'], ['p2', 'place']), live({ place: ['p1'] }));
    expect(out).toEqual(['p2']);
  });

  it('the real shape: three stale “Payments API” rows from three regenerations, all dropped at once', () => {
    const stale = rows(['old-1', 'place'], ['old-2', 'place'], ['old-3', 'place'], ['current', 'place']);
    expect(deadNodes(stale, live({ place: ['current'] }))).toEqual(['old-1', 'old-2', 'old-3']);
  });

  it('a kind we enumerated and found EMPTY does prune — we looked, and there is nothing left', () => {
    expect(deadNodes(rows(['r1', 'region']), live({ region: [] }))).toEqual(['r1']);
  });
});

describe('what must NEVER be pruned — the rule that protects the graph', () => {
  it('a kind that was never enumerated is left alone, even with no live ids', () => {
    // a retired legacy collection, or a table this build no longer loads. Treating "no live ids" as
    // "everything is dead" is how a schema change quietly deletes the whole index.
    expect(deadNodes(rows(['x1', 'location'], ['x2', 'region']), live({ place: ['p1'] }))).toEqual([]);
  });

  it('detail nodes are never this function’s business — they reconcile with their parent', () => {
    expect(deadNodes(rows(['p1#dabc', 'detail']), live({ detail: [] }))).toEqual([]);
  });

  it('an empty index prunes nothing rather than throwing', () => {
    expect(deadNodes([], live({ place: ['p1'] }))).toEqual([]);
  });

  it('no live map at all prunes nothing — a caller that failed to enumerate must not wipe the graph', () => {
    expect(deadNodes(rows(['p1', 'place'], ['e1', 'entity']), new Map())).toEqual([]);
  });

  it('kinds are independent — a dead place never takes a live entity with it', () => {
    const out = deadNodes(rows(['p1', 'place'], ['e1', 'entity']), live({ place: [], entity: ['e1'] }));
    expect(out).toEqual(['p1']);
  });
});

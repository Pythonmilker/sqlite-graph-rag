import { describe, it, expect } from 'vitest';
import Database from 'better-sqlite3';
import { betterSqlite3Driver } from './sqlite';
import { GraphRagIndex, type RecordCollections, type SourceSpec } from './index';

/**
 * TWO INVARIANTS, pinned rather than assumed.
 *
 * 1. A collection with no `SourceSpec` is UNREACHABLE — not merely unranked. Applications routinely
 *    hold data that must never be retrievable: private drafts, soft-deleted rows, another tenant's
 *    records. Relying on such rows simply scoring badly is not a guarantee; anything in the index can
 *    surface for some query. The guarantee has to be that they were never indexed, and that is worth
 *    a test because it is an ABSENCE, and absences are what silently stop being true.
 *
 * 2. `decorate` output is part of the indexed text, so a record's state can change how it retrieves.
 *    Without it, a resolved item and a live one with identical prose are indistinguishable to the
 *    index, and the model is handed a closed matter as though it were open.
 */

const uuid = (n: number) => `00000000-0000-0000-0000-${String(n).padStart(12, '0')}`;
const scopeId = uuid(1);
const PUBLISHED = uuid(30);
const DRAFT = uuid(31);
const CLOSED_TICKET = uuid(32);

/** `drafts` is deliberately absent. Its rows must never become retrievable. */
const SOURCES: SourceSpec[] = [
  { key: 'articles', kind: 'article', title: 'title', body: ['body'] },
  {
    key: 'tickets',
    kind: 'ticket',
    title: 'title',
    body: ['body'],
    decorate: (r) => (r.status === 'closed' ? 'already resolved' : ''),
  },
];

function collections(): RecordCollections {
  return {
    articles: [{ id: PUBLISHED, title: 'Rate limiting', body: 'a quixotic approach to token buckets' }],
    // Unique token so a retrieval hit would be unambiguous rather than a coincidence.
    drafts: [{ id: DRAFT, title: 'Unfinished', body: 'a zanzibar clockwork idea, not ready to publish' }],
    tickets: [{ id: CLOSED_TICKET, title: 'Login latency', body: 'p99 spiked on the auth path', status: 'closed' }],
  };
}

function index() {
  const db = betterSqlite3Driver(new Database(':memory:'));
  const kb = new GraphRagIndex({ db, embedder: null, scopeId, model: 'keyword', sources: SOURCES });
  return { db, kb };
}

describe('source scoping', () => {
  it('never indexes a collection that has no source spec', async () => {
    const { db, kb } = index();
    await kb.indexCollections(collections());

    const row = db.prepare(`SELECT 1 FROM nodes WHERE node_id = ?`).get(DRAFT);
    expect(row).toBeUndefined();
    expect(await kb.search('zanzibar')).toHaveLength(0);
  });

  it('does index the collections that are listed', async () => {
    const { kb } = index();
    await kb.indexCollections(collections());

    const hits = await kb.search('quixotic');
    expect(hits.some((h) => h.id === PUBLISHED)).toBe(true);
  });

  it('folds decorate output into the indexed body so state affects retrieval', async () => {
    const { db, kb } = index();
    await kb.indexCollections(collections());

    const row = db.prepare(`SELECT body FROM nodes WHERE node_id = ?`).get(CLOSED_TICKET) as
      | { body: string }
      | undefined;
    expect(row?.body).toContain('already resolved');
    expect(row?.body).toContain('p99 spiked on the auth path');
  });
});

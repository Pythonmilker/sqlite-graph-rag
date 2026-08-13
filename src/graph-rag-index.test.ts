import { describe, it, expect } from 'vitest';
import Database from 'better-sqlite3';
import { betterSqlite3Driver, type SqliteDb } from './sqlite';
import {
  GraphRagIndex,
  BruteForceIndex,
  type Embedder,
  type GraphEdge,
  type RecordCollections,
  type SourceSpec,
} from './index';

const uuid = (n: number) => `00000000-0000-0000-0000-${String(n).padStart(12, '0')}`;
const scopeId = uuid(1);
const AUTH = uuid(10);
const BILLING = uuid(11);
const RUNBOOK = uuid(12);
const PLATFORM = uuid(13);

/**
 * Deterministic, offline embedder: hashed bag-of-words. Shared tokens → higher cosine, so the vector
 * leg is meaningful without a network or a model. Real wiring injects any
 * `(texts) => Promise<number[][]>` — the same shape every provider's embed call already has.
 */
function hashingEmbedder(dim = 128): Embedder {
  return {
    async embed(texts: string[]): Promise<number[][]> {
      return texts.map((t) => {
        const v = new Array<number>(dim).fill(0);
        for (const tok of t.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
          let h = 2166136261;
          for (let i = 0; i < tok.length; i++) {
            h ^= tok.charCodeAt(i);
            h = Math.imul(h, 16777619);
          }
          v[Math.abs(h) % dim] += 1;
        }
        return v;
      });
    },
  };
}

/** How to read the fixture below. In real use this describes your own tables. */
const SOURCES: SourceSpec[] = [
  { key: 'services', kind: 'service', title: 'name', body: ['description'] },
  { key: 'documents', kind: 'doc', title: 'title', body: ['body'] },
  { key: 'teams', kind: 'team', title: 'name', body: ['overview'] },
];

/**
 * A small system knowledge base: two services, the runbook that documents them, and a team.
 *
 * Note what the bodies are chosen to prove. "validates" appears only in a BODY, never a title, so a
 * hit on it can only have come through the FTS leg. And the query in the stemming test is the
 * singular, which only matches because the index tokenizes with porter.
 */
function sampleCollections(): RecordCollections {
  return {
    services: [
      { id: AUTH, name: 'auth-service', description: 'Issues session tokens and validates every incoming request' },
      { id: BILLING, name: 'billing-service', description: 'Charges customers and reconciles invoices nightly' },
    ],
    documents: [{ id: RUNBOOK, title: 'Incident runbook', body: 'The on-call procedure for a paging incident' }],
    teams: [{ id: PLATFORM, name: 'Platform team', overview: 'Owns the shared infrastructure' }],
  };
}

/**
 * The `edges` table this library READS but does not own — in real use it belongs to the host
 * application. Created here with the real SQL rather than mocked, so the tests exercise the same
 * `SELECT data FROM edges` path production does.
 */
function createEdgeTable(db: SqliteDb): void {
  db.exec(`CREATE TABLE IF NOT EXISTS edges (id TEXT PRIMARY KEY, scope_id TEXT NOT NULL, data TEXT NOT NULL)`);
}

function addEdge(db: SqliteDb, edge: GraphEdge): void {
  db.prepare(`INSERT OR REPLACE INTO edges(id, scope_id, data) VALUES(?, ?, ?)`).run(
    edge.id,
    scopeId,
    JSON.stringify(edge),
  );
}

const EDGES: GraphEdge[] = [
  { id: uuid(20), sourceId: AUTH, targetId: RUNBOOK, type: 'documented_by', weight: 0.8 },
  { id: uuid(21), sourceId: BILLING, targetId: RUNBOOK, type: 'documented_by', weight: 0.9 },
];

/** Fresh in-memory DB with the edge table populated and the collections indexed. */
async function setup(): Promise<{ db: SqliteDb; kb: GraphRagIndex }> {
  const db = betterSqlite3Driver(new Database(':memory:'));
  createEdgeTable(db);
  for (const e of EDGES) addEdge(db, e);
  const kb = new GraphRagIndex({ db, embedder: hashingEmbedder(), scopeId, model: 'test-embed', sources: SOURCES });
  await kb.indexCollections(sampleCollections());
  return { db, kb };
}

describe('BruteForceIndex', () => {
  it('ranks by cosine distance, nearest first', () => {
    const idx = new BruteForceIndex();
    idx.upsert('a', Float32Array.from([1, 0]));
    idx.upsert('b', Float32Array.from([0.9, 0.1]));
    idx.upsert('c', Float32Array.from([0, 1]));
    const ranked = idx.search(Float32Array.from([1, 0]), 3).map((r) => r.id);
    expect(ranked[0]).toBe('a');
    expect(ranked[1]).toBe('b');
    expect(ranked[2]).toBe('c');
  });
});

describe('GraphRagIndex', () => {
  it('indexes every collection named by a source spec', async () => {
    const { kb } = await setup();
    expect(kb.stats()).toEqual({ nodes: 4, vectors: 4 });
  });

  it('hybrid search ranks the billing service first for "billing"', async () => {
    const { kb } = await setup();
    const hits = await kb.search('billing');
    expect(hits[0].id).toBe(BILLING);
    expect(typeof hits[0].score).toBe('number');
    expect(hits[0].snippet.length).toBeGreaterThan(0);
  });

  it('keyword path finds a body-only term ("validates")', async () => {
    const { kb } = await setup();
    const hits = await kb.search('validates');
    expect(hits.some((h) => h.id === AUTH)).toBe(true);
  });

  it('finds the runbook via "incident"', async () => {
    const { kb } = await setup();
    const hits = await kb.search('incident');
    expect(hits.some((h) => h.id === RUNBOOK)).toBe(true);
  });

  it('neighbors traverses the edge table in both directions', async () => {
    const { kb } = await setup();
    const fromAuth = kb.neighbors(AUTH);
    expect(fromAuth).toHaveLength(1);
    expect(fromAuth[0]).toMatchObject({
      id: RUNBOOK,
      relation: 'documented_by',
      direction: 'out',
      title: 'Incident runbook',
    });

    const fromRunbook = kb.neighbors(RUNBOOK);
    expect(new Set(fromRunbook.map((n) => n.id))).toEqual(new Set([AUTH, BILLING]));
    expect(fromRunbook.every((n) => n.direction === 'in')).toBe(true);
  });

  it('retrieveContext returns seeds plus their graph neighbourhood', async () => {
    const { kb } = await setup();
    const ctx = await kb.retrieveContext('billing invoices');
    expect(ctx.seeds.some((s) => s.id === BILLING)).toBe(true);
    expect(ctx.related.some((r) => r.id === RUNBOOK)).toBe(true);
  });

  it('removeNode drops a node from search', async () => {
    const { kb } = await setup();
    kb.removeNode(BILLING);
    expect(kb.stats().nodes).toBe(3);
    const hits = await kb.search('billing');
    expect(hits.some((h) => h.id === BILLING)).toBe(false);
  });

  it('rehydrates the vector index from persisted embeddings', async () => {
    const { db, kb } = await setup();
    expect(kb.stats().vectors).toBe(4);
    // A NEW instance over the same DB, with NO re-indexing, must still search.
    const reopened = new GraphRagIndex({ db, embedder: hashingEmbedder(), scopeId, sources: SOURCES });
    expect(reopened.stats().vectors).toBe(4);
    const hits = await reopened.search('billing');
    expect(hits[0].id).toBe(BILLING);
  });

  it('re-indexing a node updates it in place (no duplicates)', async () => {
    const { kb } = await setup();
    await kb.indexNode({ id: BILLING, kind: 'service', title: 'billing-service', body: 'Now emits ledger events' });
    expect(kb.stats().nodes).toBe(4);
    const hits = await kb.search('ledger');
    expect(hits.some((h) => h.id === BILLING)).toBe(true);
  });

  it('search on an empty corpus returns nothing', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    createEdgeTable(db);
    const kb = new GraphRagIndex({ db, embedder: hashingEmbedder(), scopeId, sources: SOURCES });
    expect(await kb.search('anything')).toEqual([]);
    expect(kb.neighbors(AUTH)).toEqual([]);
  });

  it('keyword search stems morphology ("validate" matches "validates")', async () => {
    const { kb } = await setup();
    // The body says "validates"; the query is the singular. Porter stemming bridges them.
    const hits = await kb.search('validate');
    expect(hits.some((h) => h.id === AUTH)).toBe(true);
  });

  it('keyword-only mode (no embedder) indexes and retrieves via FTS with stemming', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    createEdgeTable(db);
    for (const e of EDGES) addEdge(db, e);
    const kb = new GraphRagIndex({ db, embedder: null, scopeId, model: 'fts-keyword', sources: SOURCES });
    await kb.indexCollections(sampleCollections());
    expect(kb.stats()).toEqual({ nodes: 4, vectors: 0 }); // rows persist, no vector leg
    // bm25 + porter stemming still work: the singular query hits the "validates" body
    const hits = await kb.search('validate');
    expect(hits.some((h) => h.id === AUTH)).toBe(true);
    // graph expansion is untouched by the absence of an embedder
    expect(kb.neighbors(AUTH).some((n) => n.id === RUNBOOK)).toBe(true);
    // and a full retrieval works end to end without ever embedding
    const ctx = await kb.retrieveContext('validate request');
    expect(ctx.seeds.length).toBeGreaterThan(0);
  });

  it('keyword-only re-index of an embedded row drops its stale vector but keeps FTS current', async () => {
    const { db } = await setup(); // embedded: 4 nodes / 4 vectors
    const kw = new GraphRagIndex({ db, embedder: null, scopeId, model: 'fts-keyword', sources: SOURCES });
    await kw.indexNode({ id: BILLING, kind: 'service', title: 'billing-service', body: 'Now emits ledger events' });
    expect(kw.stats().nodes).toBe(4); // in-place update, no duplicate
    const hits = await kw.search('ledger');
    expect(hits.some((h) => h.id === BILLING)).toBe(true);
  });

  it('enforces the embedder contract (throws when it returns too few vectors)', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    createEdgeTable(db);
    const shortEmbedder: Embedder = {
      async embed() {
        return [];
      },
    }; // returns 0 for >0 inputs
    const kb = new GraphRagIndex({ db, embedder: shortEmbedder, scopeId, sources: SOURCES });
    await expect(kb.indexNode({ id: AUTH, kind: 'service', title: 'auth-service', body: 'tokens' })).rejects.toThrow(
      /vectors for/,
    );
    expect(kb.stats().nodes).toBe(0); // the transaction never ran — nothing persisted
  });

  it('neighbors reflects edges added after construction', async () => {
    const { db, kb } = await setup();
    // Edges are read fresh on every expansion, so a row written later is picked up without
    // reconstructing the index — the host app owns that table and writes to it on its own schedule.
    addEdge(db, { id: uuid(22), sourceId: AUTH, targetId: BILLING, type: 'calls', weight: 0.6 });
    expect(kb.neighbors(AUTH).some((n) => n.id === BILLING && n.relation === 'calls')).toBe(true);
  });

  it('clear() wipes the scope index', async () => {
    const { kb } = await setup();
    expect(kb.stats().nodes).toBe(4);
    kb.clear();
    expect(kb.stats()).toEqual({ nodes: 0, vectors: 0 });
    expect(await kb.search('billing')).toEqual([]);
  });
});

describe('host-contract hardening (regressions from review)', () => {
  const CHILD_SOURCES: SourceSpec[] = [
    { key: 'records', kind: 'record', title: 'name', body: ['description'], children: 'facts' },
  ];

  it('childNodeIdsOf treats LIKE metachars in a record id as literals, not wildcards', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    const kb = new GraphRagIndex({ db, embedder: null, scopeId, sources: CHILD_SOURCES });
    await kb.indexCollections({
      records: [
        { id: 'item_1', name: 'Underscore item', description: 'x', facts: [{ id: 'a', text: 'fact of item_1' }] },
        { id: 'itemA1', name: 'Lookalike item', description: 'y', facts: [{ id: 'b', text: 'fact of itemA1' }] },
      ],
    });
    // '_' must NOT match the 'A': returning itemA1's child here feeds the caller's delete-reconcile
    // flow another record's live children.
    expect(kb.childNodeIdsOf('item_1')).toEqual(['item_1#da']);
  });

  it('a record with no usable id is a loud error naming the field, never a silent collapse', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    const kb = new GraphRagIndex({ db, embedder: null, scopeId, sources: SOURCES });
    await expect(
      kb.indexCollections({ services: [{ _id: 'svc-1', name: 'auth-service', description: 'tokens' }] }),
    ).rejects.toThrow(/SourceSpec.id/);
    expect(kb.stats().nodes).toBe(0);
  });

  it('SourceSpec.id remaps the id field for hosts that do not key records by `id`', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    const kb = new GraphRagIndex({
      db,
      embedder: null,
      scopeId,
      sources: [{ key: 'services', kind: 'service', id: '_id', title: 'name', body: ['description'] }],
    });
    await kb.indexCollections({ services: [{ _id: 'svc-1', name: 'auth-service', description: 'issues tokens' }] });
    const hits = await kb.search('tokens');
    expect(hits.some((h) => h.id === 'svc-1')).toBe(true);
  });

  it('indexCollections with no sources throws instead of silently indexing nothing', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    const kb = new GraphRagIndex({ db, embedder: null, scopeId });
    await expect(kb.indexCollections(sampleCollections())).rejects.toThrow(/sources/);
  });

  it('an edges table created AFTER construction is picked up — the host owns its migration order', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    const kb = new GraphRagIndex({ db, embedder: hashingEmbedder(), scopeId, sources: SOURCES });
    await kb.indexCollections(sampleCollections());
    expect(kb.neighbors(AUTH)).toEqual([]); // no table yet — degrade, don't throw
    createEdgeTable(db);
    for (const e of EDGES) addEdge(db, e);
    expect(kb.neighbors(AUTH).some((n) => n.id === RUNBOOK)).toBe(true);
  });

  it('edge rows with the wrong shape are skipped; optional fields degrade instead of poisoning output', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    createEdgeTable(db);
    // wrong endpoint field names: valid JSON, wrong shape → skipped like malformed JSON
    db.prepare(`INSERT INTO edges(id, scope_id, data) VALUES('e1', ?, ?)`).run(
      scopeId,
      JSON.stringify({ id: 'e1', source_id: AUTH, target_id: RUNBOOK, type: 'documented_by', weight: 1 }),
    );
    // right endpoints, missing type/weight → kept with safe degraded values
    db.prepare(`INSERT INTO edges(id, scope_id, data) VALUES('e2', ?, ?)`).run(
      scopeId,
      JSON.stringify({ id: 'e2', sourceId: AUTH, targetId: RUNBOOK }),
    );
    const kb = new GraphRagIndex({ db, embedder: hashingEmbedder(), scopeId, sources: SOURCES });
    await kb.indexCollections(sampleCollections());
    const nbs = kb.neighbors(AUTH);
    expect(nbs).toHaveLength(1);
    expect(nbs[0]).toMatchObject({ id: RUNBOOK, relation: '', weight: 0 });
  });

  it('child expansion is opt-in, and an unmarked child fact defaults to incidental', async () => {
    const db = betterSqlite3Driver(new Database(':memory:'));
    // No `children` in the spec: an unrelated `details` array must NOT be indexed.
    const noOptIn = new GraphRagIndex({
      db,
      embedder: null,
      scopeId,
      sources: [{ key: 'orders', kind: 'order', title: 'name', body: ['description'] }],
    });
    await noOptIn.indexCollections({
      orders: [{ id: 'o1', name: 'Order 7', description: 'shipped', details: [{ id: 'd1', text: 'gift wrap, no receipt' }] }],
    });
    expect(noOptIn.stats().nodes).toBe(1);
    expect(await noOptIn.search('gift')).toHaveLength(0);

    // Opted in: the child indexes as its own node, and with no salience set it lands INCIDENTAL —
    // remembered without being volunteered; promotion is an explicit act.
    const db2 = betterSqlite3Driver(new Database(':memory:'));
    const optIn = new GraphRagIndex({
      db: db2,
      embedder: null,
      scopeId,
      sources: [{ key: 'orders', kind: 'order', title: 'name', body: ['description'], children: 'details' }],
    });
    await optIn.indexCollections({
      orders: [
        {
          id: 'o1',
          name: 'Order 7',
          description: 'shipped',
          details: [
            { id: 'd1', text: 'gift wrap, no receipt' },
            { id: 'd2', text: 12345 }, // non-string text: skipped, never a mid-transaction throw
          ],
        },
      ],
    });
    expect(optIn.stats().nodes).toBe(2);
    const [child] = await optIn.search('gift');
    expect(child.id).toBe('o1#dd1');
    expect(child.salience).toBe('incidental');
  });
});

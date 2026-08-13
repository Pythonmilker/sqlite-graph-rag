import { describe, it, expect, beforeEach } from 'vitest';
import Database from 'better-sqlite3';
import { betterSqlite3Driver, type SqliteDb } from './sqlite';
import type { Salience } from './salience';
import { GraphRagIndex, type Hit } from './graph-rag-index';
import { buildContext } from './context-builder';

/**
 * KG MEMORY (#205) — the two halves of tiering, tested where they actually bite:
 *   1. salience survives the round-trip through sqlite WITHOUT entering the searchable text, and an
 *      existing index migrates in place;
 *   2. the context block orders by tier, tags the two that carry meaning, and states the rule exactly
 *      when a tag is present — inside the token ceiling it already promises.
 */

const SCOPE = '11111111-1111-4111-8111-111111111111';

const hit = (id: string, kind: string, title: string, snippet = '', salience: Salience = 'notable'): Hit =>
  ({ id, kind, title, snippet, score: 1, salience });

describe('salience rides its own column, not the prose', () => {
  let db: SqliteDb;
  let lg: GraphRagIndex;

  beforeEach(() => {
    db = betterSqlite3Driver(new Database(':memory:'));
    lg = new GraphRagIndex({ db, embedder: null, scopeId: SCOPE });
  });

  it('round-trips every tier through search()', async () => {
    await lg.indexNodes([
      { id: 'a', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch', salience: 'defining' },
      { id: 'b', kind: 'place', title: 'Search Service', body: 'rebuilds the nightly search index', salience: 'incidental' },
      { id: 'c', kind: 'place', title: 'Auth Service', body: 'rotates the nightly signing key' },
    ]);
    const byId = new Map((await lg.search('nightly')).map((h) => [h.id, h.salience]));
    expect(byId.get('a')).toBe('defining');
    expect(byId.get('b')).toBe('incidental');
    expect(byId.get('c')).toBe('notable'); // omitted ⇒ the default
  });

  it('the tier is NOT searchable text — keyword-only mode runs on FTS alone, and a label would match itself', async () => {
    await lg.indexNodes([{ id: 'a', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch', salience: 'defining' }]);
    expect(await lg.search('defining')).toHaveLength(0);
    expect(await lg.search('incidental')).toHaveLength(0);
    expect(await lg.search('nightly')).toHaveLength(1); // …while the real prose still matches
  });

  it('re-tiering does not disturb the indexed text', async () => {
    await lg.indexNodes([{ id: 'a', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch', salience: 'notable' }]);
    await lg.indexNodes([{ id: 'a', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch', salience: 'defining' }]);
    const [h] = await lg.search('nightly');
    expect(h.snippet).toBe('runs the nightly charge batch');
    expect(h.salience).toBe('defining');
  });

  it('MIGRATES an index built before the column existed, leaving its rows readable as notable', async () => {
    // simulate the pre-#205 table exactly, rows and all, then let the service open it
    const legacy = betterSqlite3Driver(new Database(':memory:'));
    legacy.exec(`
      CREATE TABLE nodes (rowid INTEGER PRIMARY KEY, node_id TEXT NOT NULL UNIQUE, scope_id TEXT NOT NULL,
        kind TEXT NOT NULL, ref_id TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '', dim INTEGER NOT NULL DEFAULT 0, embedding BLOB, updated_at TEXT NOT NULL DEFAULT '');
      CREATE VIRTUAL TABLE nodes_fts USING fts5(title, body, tokenize = 'porter');
      INSERT INTO nodes(rowid, node_id, scope_id, kind, ref_id, title, body)
        VALUES(1, 'old', '${SCOPE}', 'place', 'old', 'Payments API', 'runs the nightly charge batch');
      INSERT INTO nodes_fts(rowid, title, body) VALUES(1, 'Payments API', 'runs the nightly charge batch');
    `);
    const svc = new GraphRagIndex({ db: legacy, embedder: null, scopeId: SCOPE });
    const [h] = await svc.search('nightly');
    expect(h.id).toBe('old');
    expect(h.salience).toBe('notable');
    // and opening it a second time must not fail on a duplicate column
    expect(() => new GraphRagIndex({ db: legacy, embedder: null, scopeId: SCOPE })).not.toThrow();
  });
});

describe('remembered details are their own retrievable nodes', () => {
  let lg: GraphRagIndex;

  beforeEach(() => {
    lg = new GraphRagIndex({ db: betterSqlite3Driver(new Database(':memory:')), embedder: null, scopeId: SCOPE });
  });

  it('finds an invented detail the parent description never mentions — the whole point of #205', async () => {
    await lg.indexNodes([
      { id: 'p', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch' },
      { id: 'p#dx', kind: 'detail', title: 'Payments API', body: 'retries failed webhook deliveries', salience: 'incidental' },
    ]);
    const hits = await lg.search('retries');
    expect(hits.map((h) => h.id)).toContain('p#dx');
    expect(hits.find((h) => h.id === 'p#dx')?.salience).toBe('incidental');
  });

  it('a detail node reports its parent as its title, so a hit arrives anchored', async () => {
    await lg.indexNodes([{ id: 'p#dx', kind: 'detail', title: 'Payments API', body: 'retries failed webhooks' }]);
    expect((await lg.search('retries'))[0].title).toBe('Payments API');
  });

  it('childNodeIdsOf lists exactly one record’s details — the set the pruner diffs against', async () => {
    await lg.indexNodes([
      { id: 'p#d1', kind: 'detail', title: 'A', body: 'one' },
      { id: 'p#d2', kind: 'detail', title: 'A', body: 'two' },
      { id: 'q#d1', kind: 'detail', title: 'B', body: 'three' },
      { id: 'p', kind: 'place', title: 'A', body: 'the place itself' },
    ]);
    expect(lg.childNodeIdsOf('p').sort()).toEqual(['p#d1', 'p#d2']);
    expect(lg.childNodeIdsOf('q')).toEqual(['q#d1']);
  });

  it('a REMOVED detail stops being retrievable — a deleted fact that still surfaces is worse than no memory', async () => {
    await lg.indexNodes([{ id: 'p#d1', kind: 'detail', title: 'A', body: 'retries failed webhooks' }]);
    expect(await lg.search('retries')).toHaveLength(1);
    lg.removeNode('p#d1');
    expect(await lg.search('retries')).toHaveLength(0);
    expect(lg.childNodeIdsOf('p')).toEqual([]);
  });

  it('details never claim a guaranteed name-seed — they borrow the parent’s title and would flood it', async () => {
    await lg.indexNodes([
      { id: 'p', kind: 'place', title: 'Payments API', body: 'runs the nightly charge batch' },
      { id: 'p#d1', kind: 'detail', title: 'Payments API', body: 'one' },
      { id: 'p#d2', kind: 'detail', title: 'Payments API', body: 'two' },
      { id: 'p#d3', kind: 'detail', title: 'Payments API', body: 'three' },
    ]);
    const ctx = await lg.retrieveContext('what do I know about Payments API?', { seeds: 2 });
    const exact = ctx.seeds.filter((s) => s.score === Infinity);
    expect(exact).toHaveLength(1);
    expect(exact[0].id).toBe('p'); // the RECORD, never its details
  });
});

describe('the context block carries the tier to the model', () => {
  it('tags the two tiers that mean something and leaves the default bare', () => {
    const r = buildContext({
      query: 'q',
      related: [],
      seeds: [
        hit('1', 'place', 'Payments API', 'runs the nightly charge batch', 'defining'),
        hit('2', 'detail', 'Payments API', 'retries failed webhooks', 'incidental'),
        hit('3', 'entity', 'Kestros', 'the keeper', 'notable'),
      ],
    });
    expect(r.text).toContain('[place · defining]');
    expect(r.text).toContain('[detail · incidental]');
    expect(r.text).toContain('[entity]'); // notable renders as no tag at all
    expect(r.text).not.toContain('· notable');
  });

  it('states the rule when a tag is present — the column alone only reorders lines', () => {
    const tagged = buildContext({ query: 'q', related: [], seeds: [hit('1', 'place', 'X', 'y', 'incidental')] });
    expect(tagged.text).toMatch(/do NOT volunteer it/);
    expect(tagged.text).toMatch(/permission to KNOW it, not an instruction to say it/);
  });

  it('drops the RULE rather than the FACTS when the ceiling is brutal', () => {
    // an empty block is the one outcome worse than an unexplained tag, so the facts are admitted first
    // and the rule is taken only if it still fits.
    const r = buildContext(
      { query: 'q', related: [], seeds: [hit('1', 'place', 'Payments API', 'runs the nightly charge batch', 'defining')] },
      // room for the header + the fact line, but not for the ~60-token rule on top
      { budgetTokens: 40, maxTokens: 40 },
    );
    expect(r.text).toContain('Payments API');
    expect(r.text).not.toMatch(/volunteer/);
    expect(r.totalTokens).toBeLessThanOrEqual(40);
  });

  it('…and stays silent when nothing is tagged, so an ordinary turn pays nothing', () => {
    const plain = buildContext({ query: 'q', related: [], seeds: [hit('1', 'place', 'X', 'y')] });
    expect(plain.text).not.toMatch(/volunteer/);
    expect(plain.text).toContain('- X [place]');
  });

  it('orders defining before notable before incidental, keeping retrieval order WITHIN a tier', () => {
    const r = buildContext({
      query: 'q',
      related: [],
      seeds: [
        hit('1', 'place', 'Incidental One', 'a', 'incidental'),
        hit('2', 'place', 'Notable One', 'b', 'notable'),
        hit('3', 'place', 'Defining One', 'c', 'defining'),
        hit('4', 'place', 'Incidental Two', 'd', 'incidental'),
      ],
    });
    const order = ['Defining One', 'Notable One', 'Incidental One', 'Incidental Two'];
    const positions = order.map((n) => r.text.indexOf(n));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  it('DEFINING facts outlive incidental ones when the budget forces a cut', () => {
    const long = 'x'.repeat(400);
    const seeds = [
      ...Array.from({ length: 20 }, (_, i) => hit(`i${i}`, 'place', `Incidental ${i} ${long}`, long, 'incidental')),
      hit('keep', 'place', 'The Defining One', 'matters', 'defining'),
    ];
    const r = buildContext({ query: 'q', related: [], seeds }, { budgetTokens: 60, maxTokens: 120 });
    expect(r.text).toContain('The Defining One');
    expect(r.totalTokens).toBeLessThanOrEqual(120);
  });

  it('NEVER breaks the hard ceiling, rule text included', () => {
    // the rule is a block of instruction prose; appending it after the accounting would silently
    // blow the ceiling this module exists to guarantee.
    for (const max of [40, 80, 160, 400]) {
      const seeds = Array.from({ length: 30 }, (_, i) =>
        hit(`n${i}`, 'place', `Place Number ${i}`, 'y'.repeat(200), i % 2 ? 'incidental' : 'defining'),
      );
      const r = buildContext({ query: 'q', related: [], seeds }, { budgetTokens: max, maxTokens: max });
      expect(r.totalTokens, `ceiling ${max}`).toBeLessThanOrEqual(max);
    }
  });

  it('a graph neighbour carries its OWN tier through expansion — the tag and trim priority survive', () => {
    // The regression this pins: neighbours used to be hardcoded 'notable', so an incidental fact
    // reached through an edge lost its tag and outranked the incidental seed that surfaced it.
    const r = buildContext({
      query: 'q',
      seeds: [hit('1', 'place', 'Seed', 's', 'notable')],
      related: [{ id: '2', title: 'Neighbour', kind: 'entity', salience: 'incidental', relation: 'calls', weight: 1, via: 'e', direction: 'out' }],
    });
    expect(r.text).toContain('- Neighbour [entity · incidental]');
    expect(r.text.indexOf('Seed')).toBeLessThan(r.text.indexOf('Neighbour')); // notable seed outranks incidental neighbour
  });
});

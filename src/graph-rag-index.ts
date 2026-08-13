import type { SqliteDb } from './sqlite';
import {
  childNodesFor,
  childNodeIdPrefix,
  asSalience,
  DETAIL_KIND,
  type ChildFact,
  type Salience,
} from './salience';

import { BruteForceIndex, bufferToF32, f32ToBuffer, type VectorIndex } from './vector-index';

/** Duck-typed embedder: anything shaped `embed(texts) => Promise<number[][]>`, hosted or local. */
export interface Embedder {
  embed(texts: string[]): Promise<number[][]>;
}

/** One indexable text unit (node_id = the source record's id). */
export interface NodeInput {
  id: string;
  kind: string;
  title: string;
  body: string;
  /** KG MEMORY: how loudly this node speaks. Stored as its OWN column, deliberately kept out of `body` —
   *  the body is both the FTS text and the embedding input, and salience belongs in neither. Absent
   *  ⇒ 'notable', which is byte-for-byte today's behaviour. */
  salience?: Salience;
}

export interface Hit {
  id: string;
  kind: string;
  title: string;
  snippet: string;
  /** fused Reciprocal-Rank-Fusion score (higher = better). */
  score: number;
  vectorRank?: number;
  keywordRank?: number;
  /** KG MEMORY: always populated (legacy rows read as 'notable'), so consumers never re-parse prose. */
  salience: Salience;
}

export interface Neighbor {
  id: string;
  title: string;
  kind: string;
  /** The neighbour's own salience, read from its node row — an incidental fact reached through an
   *  edge is still incidental, and downstream ranking must see that. */
  salience: Salience;
  relation: string;
  weight: number;
  /** the GraphEdge id this neighbour was reached through. */
  via: string;
  direction: 'out' | 'in';
}

/** A GraphRAG bundle: semantic seeds + their graph neighbourhood. */
export interface GraphContext {
  query: string;
  seeds: Hit[];
  related: Neighbor[];
}

export interface GraphRagOptions {
  db: SqliteDb;
  /** null/omitted → KEYWORD-ONLY mode: FTS5 bm25 (+ graph expansion) with no vector leg
   *  Rows persist with no embedding; FTS + edges work identically. */
  embedder?: Embedder | null;
  scopeId: string;
  /** label stored alongside each vector (the embedder model). */
  model?: string;
  /** override the vector backend (defaults to BruteForceIndex). */
  vectorIndex?: VectorIndex;
  /** Reciprocal Rank Fusion constant; 60 is the de-facto default. */
  rrfK?: number;
  /** How to read your record collections. Required only if you call `indexCollections`;
   *  `indexNode`/`indexNodes` take nodes directly and need none of this. */
  sources?: readonly SourceSpec[];
}

/** Any collection-of-records object — `{ articles: [...], people: [...] }`. Keys are matched by the
 *  `SourceSpec`s you supply; anything not named by a spec is ignored. */
export type RecordCollections = Record<string, Array<Record<string, unknown>>>;

/**
 * A typed, weighted, directed relationship between two nodes.
 *
 * This library does not own the `edges` table — you do. It expects rows of
 * `(scope_id TEXT, data TEXT)` where `data` is one of these as JSON, which is the shape that lets an
 * existing application keep whatever edge schema it already has and expose a view.
 *
 * `weight` is carried through expansion untouched rather than folded into the retrieval score.
 * Relevance and relatedness are different questions, and collapsing them means a strongly-connected
 * but irrelevant node outranks a weakly-connected answer.
 */
export interface GraphEdge {
  id: string;
  sourceId: string;
  targetId: string;
  /** Free-form relation label, surfaced on the neighbour so callers can render or filter it. */
  type: string;
  weight: number;
}

/** One traversable step of an edge, as stored in the adjacency map (an edge yields two: out and in). */
type Step = { to: string; relation: string; weight: number; via: string; direction: 'out' | 'in' };

/**
 * How to turn one collection of records into indexable nodes.
 *
 * `decorate` prepends record-derived context to the indexed body — the case that motivated it was a
 * status field: a resolved item must not retrieve identically to a live one, and the cheapest way to
 * make that true is to put the status in the text the index actually reads.
 *
 * `children` names a field holding sub-facts (notes, annotations) that get indexed as their own nodes
 * rather than concatenated into the parent. A parent with forty appended notes otherwise embeds as one
 * averaged vector that matches nothing well.
 */
export interface SourceSpec {
  /** Key in the collections object. */
  key: string;
  /** Node kind recorded on every node from this collection. */
  kind: string;
  /** Field holding each record's unique id. Defaults to `id`. A record with no usable id cannot be
   *  indexed correctly (node ids are UNIQUE with upsert semantics), so a missing one is an error,
   *  never a silent collapse. */
  id?: string;
  /** Field to use as the node title. */
  title: string;
  /** Fields concatenated into the indexed body. */
  body: string[];
  decorate?: (record: Record<string, unknown>) => string;
  /** Field holding `ChildFact[]`, indexed as separate child nodes. OPT-IN: leave unset and no child
   *  expansion happens. There is deliberately no default field name — a library that silently indexes
   *  everyone's `details` arrays is indexing data nobody offered it. */
  children?: string;
}

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS nodes (
  rowid INTEGER PRIMARY KEY,
  node_id TEXT NOT NULL UNIQUE,
  scope_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  ref_id TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  dim INTEGER NOT NULL DEFAULT 0,
  embedding BLOB,
  updated_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope_id);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(title, body, tokenize = 'porter');
`;

/** Build a safe FTS5 MATCH expression: word tokens, quoted, OR-joined for recall. */
function toFtsQuery(q: string): string | null {
  const tokens = q.toLowerCase().match(/[\p{L}\p{N}]+/gu);
  if (!tokens || tokens.length === 0) return null;
  return tokens.map((t) => `"${t}"`).join(' OR ');
}

/**
 * Hybrid (vector + FTS5) retrieval over a set of records, plus graph expansion along an `edges`
 * table. Owns only its own search index (`nodes` + `nodes_fts`) and reads `edges` if you have one,
 * so it sits alongside an existing schema rather than owning it.
 *
 * The embedder is duck-typed and optional. With none, retrieval is bm25 plus graph expansion — a
 * supported mode, not a degraded one, and the reason this runs on a machine with no network.
 */
export class GraphRagIndex {
  private readonly db: SqliteDb;
  private readonly embedder: Embedder | null;
  private readonly scopeId: string;
  private readonly model: string;
  private readonly index: VectorIndex;
  private readonly rrfK: number;
  /** Sticky-true probe for the consumer-owned `edges` table. NOT readonly: the host may create the
   *  table after constructing this index (its migrations run on its own schedule), so a false probe
   *  is re-checked on use. Snapshotting the first answer forever would silently disable the graph leg. */
  private hasEdgeTable: boolean;
  private readonly sources: readonly SourceSpec[];

  constructor(opts: GraphRagOptions) {
    this.db = opts.db;
    this.embedder = opts.embedder ?? null;
    this.scopeId = opts.scopeId;
    this.model = opts.model ?? 'unknown';
    this.index = opts.vectorIndex ?? new BruteForceIndex();
    this.rrfK = opts.rrfK ?? 60;
    this.sources = opts.sources ?? [];
    this.db.exec(SCHEMA_SQL);
    this.ensureSalienceColumn();
    this.hasEdgeTable = this.tableExists('edges');
    this.hydrate();
  }

  private tableExists(name: string): boolean {
    const row = this.db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`)
      .get(name);
    return !!row;
  }

  /** True once `edges` exists, re-probing while false — this is what makes the README's "rows written
   *  after construction are picked up" true even when the TABLE itself arrives after construction. */
  private edgeTablePresent(): boolean {
    if (!this.hasEdgeTable) this.hasEdgeTable = this.tableExists('edges');
    return this.hasEdgeTable;
  }

  /**
   * KG MEMORY migration. `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new column
   * has to be added explicitly — and this index lives OUTSIDE the store's migration runner (the service
   * owns its own DDL), so the migration belongs here, in the constructor, next to the schema it amends.
   * Idempotent by inspection rather than by catching the duplicate-column error, so a genuine failure
   * still surfaces. Existing rows default to '' and read back as 'notable'.
   */
  private ensureSalienceColumn(): void {
    const cols = this.db.prepare(`PRAGMA table_info(nodes)`).all() as Array<{ name: string }>;
    if (cols.some((c) => c.name === 'salience')) return;
    this.db.exec(`ALTER TABLE nodes ADD COLUMN salience TEXT NOT NULL DEFAULT ''`);
  }

  /** Rebuild the in-memory vector index from persisted embeddings (called on construct). */
  hydrate(): void {
    const rows = this.db
      .prepare(`SELECT node_id, embedding FROM nodes WHERE scope_id = ? AND embedding IS NOT NULL`)
      .all(this.scopeId) as Array<{ node_id: string; embedding: Uint8Array }>;
    for (const r of rows) {
      if (r.embedding) this.index.upsert(r.node_id, bufferToF32(r.embedding));
    }
  }

  stats(): { nodes: number; vectors: number } {
    const row = this.db
      .prepare(`SELECT COUNT(*) AS n FROM nodes WHERE scope_id = ?`)
      .get(this.scopeId) as { n: number };
    return { nodes: row.n, vectors: this.index.size };
  }

  /** Embed + persist a batch of nodes (one embed call), updating FTS + vector index.
   *  Keyword-only mode (no embedder): rows persist with no vector — FTS alone carries retrieval. */
  async indexNodes(inputs: NodeInput[]): Promise<number> {
    const items = inputs.filter((i) => `${i.title}${i.body}`.trim().length > 0);
    if (items.length === 0) return 0;
    const vectors = this.embedder ? await this.embedder.embed(items.map((i) => `${i.title}\n${i.body}`.trim())) : null;
    // Enforce the embedder contract: one finite-numeric vector per input. A short or
    // NaN-laden return would otherwise persist zero/garbage vectors that are silently
    // unfindable (cosine→NaN/1 → filtered), creating rows that can never be retrieved.
    if (vectors && vectors.length !== items.length) {
      throw new Error(`Embedder returned ${vectors.length} vectors for ${items.length} inputs`);
    }
    const now = new Date().toISOString();

    this.db.transaction(() => {
      const upsertNode = this.db.prepare(
        `INSERT INTO nodes(node_id, scope_id, kind, ref_id, title, body, model, dim, embedding, updated_at, salience)
         VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(node_id) DO UPDATE SET
           kind = excluded.kind, ref_id = excluded.ref_id, title = excluded.title, body = excluded.body,
           model = excluded.model, dim = excluded.dim, embedding = excluded.embedding, updated_at = excluded.updated_at,
           salience = excluded.salience
         RETURNING rowid`,
      );
      const delFts = this.db.prepare(`DELETE FROM nodes_fts WHERE rowid = ?`);
      const insFts = this.db.prepare(`INSERT INTO nodes_fts(rowid, title, body) VALUES(?, ?, ?)`);

      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        let vec: Float32Array | null = null;
        if (vectors) {
          const raw = vectors[i];
          if (!Array.isArray(raw) || raw.length === 0 || !raw.every((n) => Number.isFinite(n))) {
            throw new Error(`Embedder returned an invalid vector for node ${it.id}`);
          }
          vec = Float32Array.from(raw);
        }
        const row = upsertNode.get(
          it.id,
          this.scopeId,
          it.kind,
          it.id,
          it.title,
          it.body,
          this.model,
          vec ? vec.length : 0,
          vec ? f32ToBuffer(vec) : null,
          now,
          asSalience(it.salience),
        ) as { rowid: number } | undefined;
        if (!row) throw new Error(`Failed to upsert node ${it.id}`);
        delFts.run(row.rowid);
        insFts.run(row.rowid, it.title, it.body);
        if (vec) this.index.upsert(it.id, vec);
        else this.index.remove(it.id); // text changed under keyword mode → the old vector is stale
      }
    });
    return items.length;
  }

  /** Index a single node. */
  indexNode(input: NodeInput): Promise<number> {
    return this.indexNodes([input]);
  }

  /**
   * Index every record in every collection named by the `sources` you constructed with, plus each
   * record's child facts as their own nodes.
   *
   * Re-indexing is an upsert keyed on node id, so calling this after an edit is safe and cheap for
   * everything that did not change. What it does NOT do is remove nodes for records you deleted —
   * see `childNodeIdsOf` and `removeNode` for that, and do it, because a fact that stops existing but
   * stays retrievable is worse than one that was never indexed.
   */
  async indexCollections(collections: RecordCollections): Promise<number> {
    // Misconfiguration must be loud. With no specs this method could only ever index nothing, and a
    // silent 0 is indistinguishable from "the collections were empty" — a wiring bug that would
    // otherwise be debugged as a retrieval-quality problem.
    if (this.sources.length === 0) {
      throw new Error('indexCollections requires GraphRagOptions.sources — with no specs, nothing can be indexed');
    }
    const inputs: NodeInput[] = [];
    for (const spec of this.sources) {
      const arr = collections[spec.key];
      if (!Array.isArray(arr)) continue;
      for (const rec of arr) {
        const idField = spec.id ?? 'id';
        const rawId = rec[idField];
        // node_id is UNIQUE with upsert semantics, so records with a missing id would all collapse
        // into one node named "undefined" — silently. A loud error names the actual mistake.
        if (rawId === undefined || rawId === null || String(rawId).trim() === '') {
          throw new Error(`collection '${spec.key}': record with no '${idField}' — set SourceSpec.id to the field holding your record ids`);
        }
        const id = String(rawId);
        const title = String(rec[spec.title] ?? '').trim();
        const body = [spec.decorate?.(rec) ?? '', ...spec.body.map((f) => String(rec[f] ?? ''))]
          .join(' ')
          .trim();
        if (title || body) {
          inputs.push({
            id,
            kind: spec.kind,
            title: title || spec.kind,
            body,
            salience: asSalience(rec.salience),
          });
        }
        // Child expansion is opt-in: only a field the spec names is read.
        if (spec.children) {
          inputs.push(...childNodesFor(id, title || spec.kind, rec[spec.children] as ChildFact[] | undefined));
        }
      }
    }
    return this.indexNodes(inputs);
  }

  /**
   * The child-node ids currently indexed for a record. The caller diffs this against the record's live
   * children and removes what no longer exists — a deleted child must stop being retrievable, or a user
   * deletes a fact and the system keeps surfacing it, which is worse than never having stored it.
   */
  childNodeIdsOf(recordId: string): string[] {
    const rows = this.db
      .prepare(`SELECT node_id FROM nodes WHERE scope_id = ? AND kind = ?`)
      .all(this.scopeId, DETAIL_KIND) as Array<{ node_id: string }>;
    // Prefix-compare in JS rather than SQL LIKE: `%` and `_` in a record id must not act as wildcards
    // against OTHER records' children, because the caller deletes what this returns.
    const prefix = childNodeIdPrefix(recordId);
    return rows.map((r) => r.node_id).filter((id) => id.startsWith(prefix));
  }

  removeNode(id: string): void {
    // SELECT inside the transaction so a concurrent re-index can't change the rowid
    // between lookup and delete (which would orphan an FTS row).
    this.db.transaction(() => {
      const row = this.db
        .prepare(`SELECT rowid FROM nodes WHERE node_id = ? AND scope_id = ?`)
        .get(id, this.scopeId) as { rowid: number } | undefined;
      if (row) this.db.prepare(`DELETE FROM nodes_fts WHERE rowid = ?`).run(row.rowid);
      this.db.prepare(`DELETE FROM nodes WHERE node_id = ? AND scope_id = ?`).run(id, this.scopeId);
    });
    this.index.remove(id);
  }

  /** Wipe this corpus's entire node index (nodes + FTS + vectors). Call before a full
   *  re-index — e.g. after switching embedding models (dimensions change). */
  clear(): void {
    // node_id list is still needed for the in-memory vector removals below; the FTS delete itself is
    // set-based rather than one statement per row.
    const rows = this.db
      .prepare(`SELECT node_id FROM nodes WHERE scope_id = ?`)
      .all(this.scopeId) as Array<{ node_id: string }>;
    this.db.transaction(() => {
      this.db
        .prepare(`DELETE FROM nodes_fts WHERE rowid IN (SELECT rowid FROM nodes WHERE scope_id = ?)`)
        .run(this.scopeId);
      this.db.prepare(`DELETE FROM nodes WHERE scope_id = ?`).run(this.scopeId);
    });
    for (const r of rows) this.index.remove(r.node_id);
  }

  /** Hybrid retrieval: RRF-fuse FTS bm25 ranks with vector KNN ranks. */
  async search(query: string, opts?: { limit?: number; candidates?: number }): Promise<Hit[]> {
    const limit = opts?.limit ?? 10;
    const candidates = opts?.candidates ?? Math.max(limit * 4, 20);
    const kw = this.keywordSearch(query, candidates);
    const vec = await this.vectorSearch(query, candidates);

    const kwRank = new Map(kw.map((h) => [h.id, h.rank]));
    const vecRank = new Map(vec.map((h) => [h.id, h.rank]));
    const fused = new Map<string, number>();
    const fuse = (list: Array<{ id: string; rank: number }>) => {
      for (const h of list) fused.set(h.id, (fused.get(h.id) ?? 0) + 1 / (this.rrfK + h.rank));
    };
    fuse(kw);
    fuse(vec);

    const ranked = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit);
    if (ranked.length === 0) return [];
    return this.materialize(ranked, kwRank, vecRank);
  }

  private keywordSearch(query: string, k: number): Array<{ id: string; rank: number }> {
    const match = toFtsQuery(query);
    if (!match) return [];
    try {
      const rows = this.db
        .prepare(
          `SELECT n.node_id AS node_id
           FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid
           WHERE nodes_fts MATCH ? AND n.scope_id = ?
           ORDER BY bm25(nodes_fts, 10.0, 1.0)
           LIMIT ?`,
        )
        .all(match, this.scopeId, k) as Array<{ node_id: string }>;
      return rows.map((r, i) => ({ id: r.node_id, rank: i + 1 }));
    } catch {
      return []; // malformed FTS expression → no keyword signal
    }
  }

  private async vectorSearch(query: string, k: number): Promise<Array<{ id: string; rank: number }>> {
    if (!this.embedder || this.index.size === 0) return []; // keyword-only mode has no vector leg
    const [qvec] = await this.embedder.embed([query]);
    if (!qvec) return [];
    return this.index
      .search(Float32Array.from(qvec), k)
      .filter((h) => h.distance < 1) // drop orthogonal (zero/negative cosine) vectors — not semantic hits
      .map((h, i) => ({ id: h.id, rank: i + 1 }));
  }

  private materialize(
    ranked: Array<[string, number]>,
    kwRank: Map<string, number>,
    vecRank: Map<string, number>,
  ): Hit[] {
    const ids = ranked.map((r) => r[0]);
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT node_id, kind, title, body, salience FROM nodes WHERE scope_id = ? AND node_id IN (${placeholders})`,
      )
      .all(this.scopeId, ...ids) as Array<{ node_id: string; kind: string; title: string; body: string; salience: string }>;
    const byId = new Map(rows.map((r) => [r.node_id, r]));

    const out: Hit[] = [];
    for (const [id, score] of ranked) {
      const r = byId.get(id);
      if (!r) continue;
      out.push({
        id,
        kind: r.kind,
        title: r.title,
        snippet: this.snippetOf(r.body, r.title),
        score,
        vectorRank: vecRank.get(id),
        keywordRank: kwRank.get(id),
        salience: asSalience(r.salience),
      });
    }
    return out;
  }

  /** Graph neighbourhood of a node via the consumer's `edges` table (undirected BFS to `depth`). */
  neighbors(id: string, opts?: { depth?: number; limit?: number }): Neighbor[] {
    if (!this.edgeTablePresent()) return [];
    return this.expand(this.buildAdjacency(), id, opts);
  }

  /** Read every edge for this scope and build the traversal map. Called fresh so late-written rows
   *  are always seen; retrieveContext builds it ONCE and shares it across seeds. */
  private buildAdjacency(): Map<string, Step[]> {
    const adj = new Map<string, Step[]>();
    const link = (from: string, step: Step) => {
      const a = adj.get(from) ?? [];
      a.push(step);
      adj.set(from, a);
    };
    for (const e of this.loadEdges()) {
      link(e.sourceId, { to: e.targetId, relation: e.type, weight: e.weight, via: e.id, direction: 'out' });
      link(e.targetId, { to: e.sourceId, relation: e.type, weight: e.weight, via: e.id, direction: 'in' });
    }
    return adj;
  }

  /** BFS from `id` over a prebuilt adjacency map. Split from neighbors() so a multi-seed retrieval
   *  can expand every seed against one map instead of re-reading the edge table per seed. */
  private expand(adj: Map<string, Step[]>, id: string, opts?: { depth?: number; limit?: number }): Neighbor[] {
    const depth = Math.max(1, opts?.depth ?? 1);
    const limit = opts?.limit ?? 20;

    const visited = new Set<string>([id]);
    let frontier = [id];
    const found: Step[] = [];
    for (let d = 0; d < depth && found.length < limit; d++) {
      const next: string[] = [];
      for (const node of frontier) {
        for (const step of adj.get(node) ?? []) {
          if (visited.has(step.to)) continue;
          visited.add(step.to);
          found.push(step);
          next.push(step.to);
          if (found.length >= limit) break;
        }
        if (found.length >= limit) break;
      }
      frontier = next;
    }

    const titles = this.titlesFor(found.map((f) => f.to));
    return found.slice(0, limit).map((f) => ({
      id: f.to,
      title: titles.get(f.to)?.title ?? '',
      kind: titles.get(f.to)?.kind ?? 'unknown',
      // The neighbour's own tier rides along: an incidental fact reached through an edge is still
      // incidental, and the context builder must rank and tag it as such.
      salience: titles.get(f.to)?.salience ?? 'notable',
      relation: f.relation,
      weight: f.weight,
      via: f.via,
      direction: f.direction,
    }));
  }

  /**
   * Entities whose TITLE appears verbatim (whole-token) in the query. Ranked retrieval (vector cosine /
   * bm25) can rank an explicitly-named entity below the cutoff; this guarantees a name the reader typed is
   * never missed — the load-bearing "names never get dropped" rule of the context strategy.
   */
  private exactTitleSeeds(query: string, limit: number): Hit[] {
    if (!query.trim()) return [];
    // Narrow scan on purpose: no bodies here — only the handful of MATCHED rows need one, fetched
    // below — and child nodes are excluded in SQL: they borrow their parent's title, so every child
    // of a named record would match the name and flood the guaranteed-seed slots that exist to
    // protect the RECORD itself. Children earn their place through ranked retrieval like anything else.
    const rows = this.db
      .prepare(`SELECT node_id, kind, title, salience FROM nodes WHERE scope_id = ? AND kind != ?`)
      .all(this.scopeId, DETAIL_KIND) as Array<{ node_id: string; kind: string; title: string; salience: string }>;
    const esc = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const matched: typeof rows = [];
    for (const r of rows) {
      const t = r.title.trim();
      if (t.length < 3) continue; // too-short titles ("Ox") false-match; skip
      // collapse internal whitespace runs to \s+ so a double-spaced stored title ("Payments  API") still
      // matches a normally-spaced mention — otherwise the named entity silently loses its guaranteed seed.
      const pattern = t.split(/\s+/).map(esc).join('\\s+');
      if (new RegExp(`(^|\\W)${pattern}(\\W|$)`, 'i').test(query)) {
        matched.push(r);
        if (matched.length >= limit) break;
      }
    }
    if (matched.length === 0) return [];
    const placeholders = matched.map(() => '?').join(',');
    const bodies = this.db
      .prepare(`SELECT node_id, body FROM nodes WHERE scope_id = ? AND node_id IN (${placeholders})`)
      .all(this.scopeId, ...matched.map((m) => m.node_id)) as Array<{ node_id: string; body: string }>;
    const bodyOf = new Map(bodies.map((b) => [b.node_id, b.body]));
    return matched.map((r) => ({
      id: r.node_id,
      kind: r.kind,
      title: r.title,
      snippet: this.snippetOf(bodyOf.get(r.node_id) ?? '', r.title),
      score: Infinity,
      salience: asSalience(r.salience),
    }));
  }

  /** Hybrid search → 1-hop graph expansion → a context bundle for the AI (GraphRAG). */
  async retrieveContext(
    query: string,
    opts?: { seeds?: number; depth?: number; neighbors?: number },
  ): Promise<GraphContext> {
    const want = opts?.seeds ?? 5;
    const ranked = await this.search(query, { limit: want });
    // Exact-name hits FIRST (highest authority), then the ranked hits, deduped. Named entities always survive.
    const exact = this.exactTitleSeeds(query, want);
    const seeds: Hit[] = [];
    const seenSeed = new Set<string>();
    for (const h of [...exact, ...ranked]) {
      if (seenSeed.has(h.id)) continue;
      seenSeed.add(h.id);
      seeds.push(h);
    }
    seeds.splice(Math.max(want, exact.length)); // cap, but never below the count of explicitly-named entities
    const related: Neighbor[] = [];
    const seen = new Set(seeds.map((s) => s.id));
    if (this.edgeTablePresent()) {
      // One adjacency build for the whole retrieval. Expanding through neighbors() per seed would
      // re-read and re-JSON.parse the entire edge table once per seed, for identical data.
      const adj = this.buildAdjacency();
      for (const s of seeds) {
        for (const nb of this.expand(adj, s.id, { depth: opts?.depth ?? 1, limit: opts?.neighbors ?? 5 })) {
          if (seen.has(nb.id)) continue;
          seen.add(nb.id);
          related.push(nb);
        }
      }
    }
    return { query, seeds, related };
  }

  private loadEdges(): GraphEdge[] {
    const rows = this.db
      .prepare(`SELECT data FROM edges WHERE scope_id = ?`)
      .all(this.scopeId) as Array<{ data: string }>;
    const out: GraphEdge[] = [];
    for (const r of rows) {
      try {
        const e = JSON.parse(r.data) as Partial<GraphEdge>;
        // The host owns this table, so SHAPE is untrusted exactly like syntax is: an edge whose
        // endpoints aren't strings would key the adjacency map on `undefined` and leak undefined
        // relation/weight into the Neighbor contract. A wrong-shaped row is skipped like a malformed
        // one; the two optional fields degrade to safe values instead of poisoning the output.
        if (typeof e?.id !== 'string' || typeof e.sourceId !== 'string' || typeof e.targetId !== 'string') continue;
        out.push({
          id: e.id,
          sourceId: e.sourceId,
          targetId: e.targetId,
          type: typeof e.type === 'string' ? e.type : '',
          weight: typeof e.weight === 'number' && Number.isFinite(e.weight) ? e.weight : 0,
        });
      } catch {
        // skip malformed row
      }
    }
    return out;
  }

  private titlesFor(ids: string[]): Map<string, { title: string; kind: string; salience: Salience }> {
    const uniq = [...new Set(ids)];
    if (uniq.length === 0) return new Map();
    const placeholders = uniq.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT node_id, title, kind, salience FROM nodes WHERE scope_id = ? AND node_id IN (${placeholders})`,
      )
      .all(this.scopeId, ...uniq) as Array<{ node_id: string; title: string; kind: string; salience: string }>;
    return new Map(rows.map((r) => [r.node_id, { title: r.title, kind: r.kind, salience: asSalience(r.salience) }]));
  }

  /** One home for the snippet policy, so ranked hits and exact-name seeds render identically. */
  private snippetOf(body: string, title: string): string {
    return (body || title).slice(0, 200);
  }
}

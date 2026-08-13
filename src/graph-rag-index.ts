import type { SqliteDb } from './sqlite';
import { childNodesFor, asSalience, type ChildFact, type Salience } from './salience';

import { BruteForceIndex, bufferToF32, f32ToBuffer, type VectorIndex } from './vector-index';

/** Duck-typed embedder — pass `AiService.embed` without importing @hub/ai. */
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
  /** Field to use as the node title. */
  title: string;
  /** Fields concatenated into the indexed body. */
  body: string[];
  decorate?: (record: Record<string, unknown>) => string;
  /** Field holding `ChildFact[]`. Defaults to `details`. */
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
  private readonly hasEdgeTable: boolean;
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
    const data = collections;
    const inputs: NodeInput[] = [];
    for (const spec of this.sources) {
      const arr = data[spec.key];
      if (!Array.isArray(arr)) continue;
      for (const rec of arr) {
        const title = String(rec[spec.title] ?? '').trim();
        const body = [spec.decorate?.(rec) ?? '', ...spec.body.map((f) => String(rec[f] ?? ''))]
          .join(' ')
          .trim();
        if (title || body) {
          inputs.push({
            id: String(rec.id),
            kind: spec.kind,
            title: title || spec.kind,
            body,
            salience: asSalience(rec.salience),
          });
        }
        inputs.push(...childNodesFor(String(rec.id), title || spec.kind, rec[spec.children ?? "details"] as ChildFact[] | undefined));
      }
    }
    return this.indexNodes(inputs);
  }

  /**
   * The detail-node ids currently indexed for a record. The caller diffs this against the record's live
   * details and removes what no longer exists — a deleted child must stop being retrievable, or a user
   * deletes a fact and the system keeps surfacing it, which is worse than never having stored it.
   */
  detailNodeIdsOf(recordId: string): string[] {
    const rows = this.db
      .prepare(`SELECT node_id FROM nodes WHERE scope_id = ? AND kind = 'detail' AND node_id LIKE ?`)
      .all(this.scopeId, `${recordId}#d%`) as Array<{ node_id: string }>;
    return rows.map((r) => r.node_id);
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
    const rows = this.db
      .prepare(`SELECT rowid, node_id FROM nodes WHERE scope_id = ?`)
      .all(this.scopeId) as Array<{ rowid: number; node_id: string }>;
    this.db.transaction(() => {
      const delFts = this.db.prepare(`DELETE FROM nodes_fts WHERE rowid = ?`);
      for (const r of rows) delFts.run(r.rowid);
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
        snippet: (r.body || r.title).slice(0, 200),
        score,
        vectorRank: vecRank.get(id),
        keywordRank: kwRank.get(id),
        salience: asSalience(r.salience),
      });
    }
    return out;
  }

  /** Graph neighbourhood of a node via the store's `edges` (undirected BFS to `depth`). */
  neighbors(id: string, opts?: { depth?: number; limit?: number }): Neighbor[] {
    if (!this.hasEdgeTable) return [];
    const depth = Math.max(1, opts?.depth ?? 1);
    const limit = opts?.limit ?? 20;

    type Step = { to: string; relation: string; weight: number; via: string; direction: 'out' | 'in' };
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
    const rows = this.db
      .prepare(`SELECT node_id, kind, title, body, salience FROM nodes WHERE scope_id = ?`)
      .all(this.scopeId) as Array<{ node_id: string; kind: string; title: string; body: string; salience: string }>;
    const esc = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const out: Hit[] = [];
    for (const r of rows) {
      const t = r.title.trim();
      if (t.length < 3) continue; // too-short titles ("Ox") false-match; skip
      // DETAIL nodes borrow their parent's title, so every detail of a named place would match the name and
      // flood the guaranteed-seed slots that exist to protect the RECORD itself. Details earn their place
      // through ranked retrieval like anything else.
      if (r.kind === 'detail') continue;
      // collapse internal whitespace runs to \s+ so a double-spaced stored title ("Payments  API") still
      // matches a normally-spaced mention — otherwise the named entity silently loses its guaranteed seed.
      const pattern = t.split(/\s+/).map(esc).join('\\s+');
      if (new RegExp(`(^|\\W)${pattern}(\\W|$)`, 'i').test(query)) {
        out.push({
          id: r.node_id,
          kind: r.kind,
          title: r.title,
          snippet: (r.body || r.title).slice(0, 200),
          score: Infinity,
          salience: asSalience(r.salience),
        });
        if (out.length >= limit) break;
      }
    }
    return out;
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
    for (const s of seeds) {
      for (const nb of this.neighbors(s.id, { depth: opts?.depth ?? 1, limit: opts?.neighbors ?? 5 })) {
        if (seen.has(nb.id)) continue;
        seen.add(nb.id);
        related.push(nb);
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
        out.push(JSON.parse(r.data) as GraphEdge);
      } catch {
        // skip malformed row
      }
    }
    return out;
  }

  private titlesFor(ids: string[]): Map<string, { title: string; kind: string }> {
    const uniq = [...new Set(ids)];
    if (uniq.length === 0) return new Map();
    const placeholders = uniq.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT node_id, title, kind FROM nodes WHERE scope_id = ? AND node_id IN (${placeholders})`,
      )
      .all(this.scopeId, ...uniq) as Array<{ node_id: string; title: string; kind: string }>;
    return new Map(rows.map((r) => [r.node_id, { title: r.title, kind: r.kind }]));
  }
}

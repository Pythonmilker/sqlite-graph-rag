# sqlite-graph-rag

Graph RAG that runs inside a single SQLite file. Retrieval fuses a vector search and a BM25 keyword
search, then expands along a typed edge table and returns the neighbourhood as model context.

There is no vector database, no server, and no network call. It works with no embedder at all.

This is a code sample extracted from a shipping product, not a package to install. See
[License](#license).

## How retrieval works

Three stages, and the third is what makes it graph RAG rather than vector search:

1. **Seed.** The query runs against an FTS5 index (bm25, porter stemming) and, if an embedder is
   configured, against a vector index. The two ranked lists are fused with Reciprocal Rank Fusion,
   so a result that both legs like beats one that either loves.
2. **Expand.** Each seed's neighbours are read from an `edges` table by breadth-first traversal,
   carrying the relation label, weight, and direction.
3. **Return.** Seeds plus neighbourhood, ready to be budgeted into a prompt.

Vector search alone answers "what looks like this query". The expansion step answers "and what is
this connected to", which is the half that lets a model reason about a record it was never asked
about directly.

## With no embedder

Leave `embedder` unset and the vector leg drops out. FTS5 and graph expansion carry retrieval on
their own, and `retrieveContext` still returns seeds and their neighbourhood.

This is a supported mode rather than a fallback. It is why the whole thing runs on a laptop with no
network, and it is worth knowing that graph expansion is where a keyword-only setup gets most of its
apparent intelligence.

## What you bring

This is the retrieval middle of a RAG pipeline. Four things come from the host application, and the
library stops one step short of the model:

- **Records**, as plain objects in named collections, described by `SourceSpec`s. A collection no
  spec names is never indexed, and that absence is a guarantee the tests pin.
- **The `edges` table**, if you want expansion. The host owns it and writes it; this library only
  reads it.
- **An embedder, optionally** — anything shaped `embed(texts: string[]) => Promise<number[][]>`, one
  vector per input. The product this was extracted from injects a small static Model2Vec model that
  runs on-device; the tests inject a deterministic hashing embedder; a hosted embeddings endpoint has
  the same shape. Vectors from different models share no space, so changing models means `clear()`
  and re-indexing — the `model` label stored on every node exists to catch exactly that.
- **A SQLite connection**, through the four-method interface in `src/sqlite.ts`. An adapter for
  better-sqlite3 is included.

Where it stops: `retrieveContext` returns seeds plus their neighbours, and `buildContext` renders
them into a token-budgeted prompt block. Calling a model with that block, and everything after,
belongs to the host.

## Example

```ts
import Database from 'better-sqlite3';
import { GraphRagIndex, betterSqlite3Driver } from './src';

const db = betterSqlite3Driver(new Database('app.db'));

const kb = new GraphRagIndex({
  db,
  scopeId: 'workspace-1',
  embedder: myEmbedder,          // or omit for keyword + graph only
  sources: [
    { key: 'services', kind: 'service', title: 'name', body: ['description'] },
    { key: 'documents', kind: 'doc', title: 'title', body: ['body'] },
  ],
});

await kb.indexCollections({
  services: [{ id: 'svc-1', name: 'auth-service', description: 'Issues session tokens' }],
  documents: [{ id: 'doc-1', title: 'Incident runbook', body: 'The on-call procedure' }],
});

const ctx = await kb.retrieveContext('who issues tokens');
// ctx.seeds    → ranked hits
// ctx.related  → their graph neighbours, with relation and direction
```

`sources` is the only place the library learns your schema. A collection you do not list is never
indexed and cannot be retrieved, which is the guarantee `src/source-scoping.test.ts` exists to pin.

## The edge table

The library reads `edges` and does not own it, so it sits alongside a schema you already have:

```sql
CREATE TABLE edges (id TEXT PRIMARY KEY, scope_id TEXT NOT NULL, data TEXT NOT NULL);
-- data is JSON: { id, sourceId, targetId, type, weight }
```

Edges are read fresh on every expansion, so rows written after construction are picked up without
rebuilding anything.

Weight rides through expansion untouched instead of folding into the retrieval score. Relevance and
relatedness are different questions, and collapsing them lets a strongly connected but irrelevant
node outrank the actual answer.

## Salience

Every node carries a salience of `defining`, `notable`, or `incidental`, in its own column rather
than in the indexed text. Two reasons, both found by watching retrieval run:

- With no embedder, retrieval is bm25 over the body. A label in the body becomes a searchable token,
  so a query containing "defining" would match every defining record on the strength of its own label.
- The body is the embedding input, so re-ranking a record would rewrite its text and force a re-embed.

It exists because storage and recall are different questions. A fact can be worth keeping and still
be wrong to volunteer every turn.

## Run it

Node 20 or newer.

```bash
npm install
npm test          # 55 tests
npm run typecheck # tsc --noEmit, strict
```

The tests use a deterministic hashing embedder, so the vector path is exercised with no network and
no model download.

## Layout

```
src/graph-rag-index.ts   indexing, hybrid search, expansion, retrieveContext
src/vector-index.ts      brute-force cosine index; swap via the VectorIndex interface
src/context-builder.ts   token-budgeted assembly of a retrieval result into prompt text
src/salience.ts          salience, and expanding child facts into their own nodes
src/prune.ts             which nodes outlived the record they came from
src/sqlite.ts            the four-method SQLite interface, plus a better-sqlite3 adapter
```

`better-sqlite3` is a dev dependency only. The library talks to the `SqliteDb` interface in
`src/sqlite.ts`, so another driver is an afternoon.

## Known limits

- The vector index is a brute-force cosine scan held in memory. That is deliberate at the scale this
  was built for and it is the first thing to replace if you have hundreds of thousands of nodes.
  `VectorIndex` is the seam.
- Expansion is breadth-first to a depth you pass, with no path scoring. Depth 1 or 2 is what gets
  used in practice.
- `deadNodes` decides what to prune but does not run the deletion; the caller owns that SQL.

## License

All rights reserved. This repository is published to be read, not used. No permission is granted to
copy, modify, or distribute it. It is extracted from a commercial product and shared as a work
sample.

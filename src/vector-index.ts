/**
 * Pluggable nearest-neighbour index. The hybrid retriever only needs a ranked id
 * list, so the vector backend is swappable: `BruteForceIndex` (here, dependency-free,
 * correct everywhere) is the baseline; a sqlite-vec or ANN-backed index can drop in
 * behind this same interface when corpus size demands one.
 */
export interface VectorIndex {
  upsert(id: string, vector: Float32Array): void;
  remove(id: string): void;
  /** k nearest by cosine distance, ascending (0 = identical). */
  search(query: Float32Array, k: number): Array<{ id: string; distance: number }>;
  readonly size: number;
}

/** L2-normalize a vector (returns a fresh array; zero-vector passes through). */
export function l2normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s);
  const out = new Float32Array(v.length);
  if (n === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

/**
 * In-memory cosine index: stores L2-normalized vectors so cosine similarity is a
 * single dot loop, ranked by a full linear scan. Measured 1–13ms for a personal-scale
 * corpus (≤5k vectors) on Node 24 — instant for interactive use, and GraphRagIndex
 * rehydrates it from persisted embeddings on construction.
 */
export class BruteForceIndex implements VectorIndex {
  private readonly vecs = new Map<string, Float32Array>();

  get size(): number {
    return this.vecs.size;
  }

  upsert(id: string, vector: Float32Array): void {
    this.vecs.set(id, l2normalize(vector));
  }

  remove(id: string): void {
    this.vecs.delete(id);
  }

  search(query: Float32Array, k: number): Array<{ id: string; distance: number }> {
    const q = l2normalize(query);
    const out: Array<{ id: string; distance: number }> = [];
    for (const [id, v] of this.vecs) {
      if (v.length !== q.length) continue; // dimension mismatch (model change) — skip
      let dot = 0;
      for (let i = 0; i < q.length; i++) dot += q[i] * v[i];
      out.push({ id, distance: 1 - dot });
    }
    out.sort((a, b) => a.distance - b.distance);
    return out.slice(0, k);
  }
}

/** View a Float32Array's bytes as a Buffer for BLOB binding (no copy). */
export function f32ToBuffer(v: Float32Array): Buffer {
  return Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

/** Decode a stored BLOB back to a Float32Array (copies into an aligned buffer). */
export function bufferToF32(buf: Uint8Array): Float32Array {
  if (buf.byteLength % 4 !== 0) {
    throw new Error(`Misaligned embedding BLOB: ${buf.byteLength} bytes (not a multiple of 4)`);
  }
  const ab = new ArrayBuffer(buf.byteLength);
  new Uint8Array(ab).set(buf);
  return new Float32Array(ab);
}

export { GraphRagIndex } from './graph-rag-index';
export type {
  Embedder,
  NodeInput,
  Hit,
  Neighbor,
  GraphContext,
  GraphRagOptions,
  GraphEdge,
  RecordCollections,
  SourceSpec,
} from './graph-rag-index';

export { BruteForceIndex, l2normalize } from './vector-index';
export type { VectorIndex } from './vector-index';

export { buildContext, estimateTokens } from './context-builder';
export type { ContextOptions, ContextResult } from './context-builder';

export { deadNodes } from './prune';
export type { NodeRef } from './prune';

export {
  asSalience,
  salienceTag,
  childNodesFor,
  childNodeId,
  childNodeIdPrefix,
  DEFAULT_SALIENCE,
  DETAIL_KIND,
  SALIENCE_RANK,
} from './salience';
export type { Salience, ChildFact } from './salience';

export { betterSqlite3Driver } from './sqlite';
export type { SqliteDb, SqliteStatement } from './sqlite';

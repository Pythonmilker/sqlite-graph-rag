/**
 * A minimal synchronous SQLite interface, so the retrieval logic is written once and stays
 * decoupled from any concrete driver — better-sqlite3 on a desktop, a mobile driver elsewhere.
 *
 * The interface is deliberately four methods wide. Everything this library needs is expressible
 * in `prepare` + `exec` + `transaction`, and keeping the surface that small is what makes a new
 * backend an afternoon rather than a project.
 */
export interface SqliteStatement {
  run(...params: unknown[]): void;
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}

export interface SqliteDb {
  exec(sql: string): void;
  prepare(sql: string): SqliteStatement;
  /** Run `fn` inside a transaction (commit on return, rollback on throw). */
  transaction<T>(fn: () => T): T;
  close(): void;
}

/**
 * The slice of a better-sqlite3 `Database` we use. Declared structurally rather than imported so
 * better-sqlite3 stays a peer concern: a consumer on another driver never installs it.
 *
 * Note `transaction` returns a callable (better-sqlite3's `Transaction`), which we invoke
 * immediately — the adapter exists mostly to hide that one difference.
 */
interface BetterSqlite3Like {
  exec(sql: string): unknown;
  prepare(sql: string): {
    run(...p: unknown[]): unknown;
    get(...p: unknown[]): unknown;
    all(...p: unknown[]): unknown[];
  };
  transaction(fn: () => unknown): () => unknown;
  close(): void;
}

/** Adapt a better-sqlite3 `Database` instance to the `SqliteDb` interface. */
export function betterSqlite3Driver(db: BetterSqlite3Like): SqliteDb {
  return {
    exec: (sql) => {
      db.exec(sql);
    },
    prepare: (sql) => {
      const s = db.prepare(sql);
      return {
        run: (...p) => {
          s.run(...p);
        },
        get: (...p) => s.get(...p),
        all: (...p) => s.all(...p),
      };
    },
    transaction: <T>(fn: () => T): T => (db.transaction(fn) as () => T)(),
    close: () => db.close(),
  };
}

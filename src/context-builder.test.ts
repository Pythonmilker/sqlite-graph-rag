import { describe, it, expect } from 'vitest';
import type { Salience } from './salience';
import { buildContext, estimateTokens, type ContextOptions } from './context-builder';
import type { GraphContext, Hit, Neighbor } from './graph-rag-index';

const hit = (id: string, kind: string, title: string, snippet = '', salience: Salience = 'notable'): Hit =>
  ({ id, kind, title, snippet, score: 1, salience });
const neighbor = (id: string, title: string, relation: string, kind = 'entity', salience: Salience = 'notable'): Neighbor => ({
  id,
  title,
  kind,
  salience,
  relation,
  weight: 1,
  via: `edge-${id}`,
  direction: 'out',
});
const ctx = (seeds: Hit[], related: Neighbor[] = []): GraphContext => ({ query: 'q', seeds, related });

describe('estimateTokens', () => {
  it('is 0 for empty and grows ~1 per 4 Latin chars', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('abcde')).toBe(2);
  });

  it('OVER-counts CJK so it can serve as a real ceiling (not chars/4)', () => {
    const jp = 'これはテスト'; // 6 CJK chars — chars/4 would say 2; real tokenizers emit ≥6
    expect(estimateTokens(jp)).toBeGreaterThanOrEqual(jp.length);
    const zh = '龍'.repeat(50);
    expect(estimateTokens(zh)).toBeGreaterThanOrEqual(50); // never under-reports CJK
  });
});

describe('buildContext', () => {
  it('returns an empty result for null / empty bundles', () => {
    for (const c of [null, undefined, ctx([], [])]) {
      const r = buildContext(c as GraphContext);
      expect(r.text).toBe('');
      expect(r.totalTokens).toBe(0);
      expect(r.factsCount).toBe(0);
    }
  });

  it('renders entity seeds as authoritative FACTS with name + kind + description', () => {
    const r = buildContext(ctx([hit('1', 'entity', 'auth-service', 'issues session tokens for every request')]));
    expect(r.text).toContain('FACTS (authoritative');
    expect(r.text).toContain('auth-service [entity]');
    expect(r.text).toContain('issues session tokens');
    expect(r.factsCount).toBe(1);
    expect(r.narrativeCount).toBe(0);
  });

  it('routes the kinds the CALLER names to NARRATIVE — routing is opt-in, never assumed', () => {
    const c = ctx([
      hit('1', 'entity', 'Payments API', 'reconciles invoices nightly'),
      hit('2', 'note', 'Standup 3', 'the deploy freeze lifted after the incident closed'),
      hit('3', 'update', 'Cutover plan', 'a migration pledged for the next window'),
    ]);
    const r = buildContext(c, { narrativeKinds: ['note', 'update'] });
    expect(r.text).toContain('FACTS (authoritative');
    expect(r.text).toContain('Payments API [entity]');
    expect(r.text).toContain('RECENT NARRATIVE');
    expect(r.text).toContain('deploy freeze lifted');
    expect(r.factsCount).toBe(1);
    expect(r.narrativeCount).toBe(2);
    // narrative prose must sit under the NON-authoritative header, never above the facts header
    expect(r.text.indexOf('FACTS (authoritative')).toBeLessThan(r.text.indexOf('RECENT NARRATIVE'));

    // and WITHOUT the option, the same kinds are ordinary authoritative facts: a consumer's 'note'
    // records are canon unless the consumer says otherwise
    const plain = buildContext(c);
    expect(plain.narrativeCount).toBe(0);
    expect(plain.factsCount).toBe(3);
  });

  it('renders graph neighbours as facts carrying the relation', () => {
    const r = buildContext(ctx([hit('1', 'entity', 'auth-service', 'issues session tokens')], [neighbor('2', 'Payments API', 'calls', 'location')]));
    expect(r.text).toContain('Payments API [location]');
    expect(r.text).toContain('calls');
    expect(r.factsCount).toBe(2);
  });

  it('a neighbour keeps its own tier — incidental reached through an edge is still incidental', () => {
    const r = buildContext(
      ctx([hit('1', 'entity', 'auth-service', 'issues tokens')], [neighbor('2', 'Legacy relay', 'calls', 'service', 'incidental')]),
    );
    expect(r.text).toContain('Legacy relay [service · incidental]');
    // and it loses the trim-priority contest against the notable seed
    expect(r.text.indexOf('auth-service')).toBeLessThan(r.text.indexOf('Legacy relay'));
  });

  it('NEVER drops a name under soft-budget pressure — descriptions truncate first', () => {
    const seeds = Array.from({ length: 20 }, (_, i) =>
      hit(String(i), 'entity', `Entity${i}`, 'a long winding description '.repeat(8)),
    );
    const opts: ContextOptions = { budgetTokens: 140, maxTokens: 6000 };
    const r = buildContext(ctx(seeds), opts);
    // every retrieved name survives...
    for (let i = 0; i < 20; i++) expect(r.text).toContain(`Entity${i} [entity]`);
    expect(r.factsCount).toBe(20);
    // ...but the budget forced most descriptions to be dropped
    expect(r.factsTruncated).toBeGreaterThan(0);
  });

  it('NEVER exceeds the hard ceiling, even with a flood of seeds', () => {
    const seeds = Array.from({ length: 500 }, (_, i) =>
      hit(String(i), 'entity', `VeryLongEntityName_${i}`, 'x'.repeat(400)),
    );
    const r = buildContext(ctx(seeds), { budgetTokens: 100, maxTokens: 300 });
    expect(r.totalTokens).toBeLessThanOrEqual(300);
  });

  it('is deterministic — identical input yields byte-identical text', () => {
    const c = ctx([hit('1', 'entity', 'auth-service', 'issues session tokens'), hit('2', 'note', 'note-1', 'deploy completed')]);
    expect(buildContext(c).text).toBe(buildContext(c).text);
  });

  it('totalTokens is the measured size of the assembled text', () => {
    const r = buildContext(ctx([hit('1', 'entity', 'auth-service', 'issues session tokens')]));
    expect(r.totalTokens).toBe(estimateTokens(r.text));
  });
});

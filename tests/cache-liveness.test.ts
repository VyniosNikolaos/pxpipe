/**
 * Which upstream outcomes end a session's prefix cache.
 *
 * Getting this wrong is expensive in both directions:
 *
 *   - miss a real rejection and the grid stays frozen at a shape nothing is
 *     cached against, so we keep paying for a freeze that buys nothing;
 *   - call a LIVE cache dead and the next turn re-cuts the grid, changing every
 *     chunk's bytes and burning the whole prefix as cache_create.
 *
 * The second is the worse one, so anything ambiguous must stay warm.
 *
 * Run just this file:  pnpm vitest run tests/cache-liveness.test.ts
 */
import { beforeEach, describe, expect, it } from 'vitest';
import {
  markCacheDead,
  peekSessionState,
  resetSessionState,
  responseLeftNoCache,
} from '../src/core/session-state.js';

describe('responseLeftNoCache — outcomes that leave no cache entry', () => {
  it('treats an outright size rejection as cache-ending', () => {
    expect(responseLeftNoCache(413)).toBe(true);
  });

  it('treats every 5xx as cache-ending — the over-cap image count returns an opaque 500', () => {
    for (const s of [500, 502, 503, 529]) expect(responseLeftNoCache(s)).toBe(true);
  });

  it('reads a 400 body for the provider’s several spellings of "too long"', () => {
    const bodies = [
      '{"error":{"message":"prompt is too long: 245000 tokens > 200000 maximum"}}',
      '{"error":{"type":"prompt_too_long"}}',
      '{"error":{"type":"request_too_large","message":"request too large"}}',
      '{"error":{"message":"too many images in request"}}',
    ];
    for (const b of bodies) expect(responseLeftNoCache(400, b)).toBe(true);
  });

  it('leaves the session warm on a 200', () => {
    expect(responseLeftNoCache(200)).toBe(false);
  });

  // These are the dangerous ones: all say nothing about the prefix cache, which
  // may well still be live. Re-cutting on them would burn it.
  it('leaves the session warm on 4xx that carry no size signal', () => {
    expect(responseLeftNoCache(401)).toBe(false); // bad key
    expect(responseLeftNoCache(403)).toBe(false); // forbidden
    expect(responseLeftNoCache(404)).toBe(false); // wrong path
    expect(responseLeftNoCache(429)).toBe(false); // rate limited — cache intact
    expect(responseLeftNoCache(400)).toBe(false); // 400 with no body to read
    expect(responseLeftNoCache(400, '{"error":{"message":"invalid model"}}')).toBe(false);
  });
});

describe('markCacheDead — session bookkeeping', () => {
  beforeEach(() => resetSessionState());

  it('flags the session so the next collapse may repack', () => {
    markCacheDead('abcd1234');
    expect(peekSessionState('abcd1234')?.cacheDead).toBe(true);
  });

  it('is a no-op without a session key rather than throwing', () => {
    expect(() => markCacheDead(undefined)).not.toThrow();
  });

  it('does not invent state for sessions that never failed', () => {
    markCacheDead('aaaa1111');
    expect(peekSessionState('bbbb2222')).toBeUndefined();
  });
});

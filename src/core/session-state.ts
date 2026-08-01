/**
 * Per-session cache-liveness state for the history collapse.
 *
 * ## Why this exists
 *
 * The history grid is append-only: chunk N's pixels are a pure function of its
 * message range, so old chunks stay byte-identical as the conversation grows and
 * ride Anthropic's prompt cache as `cache_read` forever. That freeze is worth a
 * lot — but only while a cache actually exists. Two situations end it:
 *
 *  1. **Idle gap.** Anthropic's ephemeral prefix cache lives {@link CACHE_TTL_SEC}
 *     seconds past the last hit. Resume a session the next morning and every
 *     block is `cache_create` again no matter what we send.
 *  2. **A rejected request.** An oversized request (opaque `500`, see
 *     {@link ANTHROPIC_MAX_IMAGES}) never populated a cache entry at all.
 *
 * In both cases the append-only freeze protects nothing, and the grid is free to
 * be re-cut for *density* instead: {@link HistoryCollapseOptions.packFill} raises
 * the freeze step until the pages are nearly full, which roughly halves image
 * tokens on long sessions of short turns (#161: 317 images at 43% fill).
 *
 * ## Why the step is sticky
 *
 * Once a session has been repacked coarse, every later turn must keep at least
 * that step. Falling back to the fine grid would re-cut the same messages into
 * different chunks — every chunk's bytes change, and the whole history re-keys as
 * `cache_create`. {@link recordFreezeStep} pins the floor; the collapse only ever
 * doubles it.
 *
 * ## Failure mode we deliberately accept
 *
 * State is in-memory and per proxy process. After a restart a live session looks
 * *unknown*, and unknown is treated as WARM (no repack) — the conservative
 * choice: at worst we keep paying the old image count, we never nuke a live cache
 * on a guess. The state re-arms itself on the first idle gap after the restart.
 */

import { CACHE_TTL_SEC } from './baseline.js';

/** Sessions tracked before the oldest is evicted. One small record each. */
const SESSIONS_MAX = 512;

/**
 * Grace added to the provider TTL before we call a cache dead. Our clock is the
 * request-arrival time, the provider's is its own; a request that lands one
 * second inside the window can still miss. Only gaps clearly past the TTL flip
 * the session cold, so a borderline case keeps the (cheap, correct) warm path.
 */
const COLD_GRACE_MS = 30_000;

interface SessionRecord {
  /** Wall-clock ms of the last request we saw for this session. */
  lastSeenMs: number;
  /** Coarsest freeze step this session has been rendered at, in messages. */
  freezeStep: number;
  /** Set when a request for this session failed in a way that leaves no cache. */
  cacheDead: boolean;
}

const sessions = new Map<string, SessionRecord>();

function touch(key: string): SessionRecord {
  const existing = sessions.get(key);
  if (existing) {
    sessions.delete(key); // refresh LRU position
    sessions.set(key, existing);
    return existing;
  }
  const fresh: SessionRecord = { lastSeenMs: 0, freezeStep: 0, cacheDead: false };
  sessions.set(key, fresh);
  while (sessions.size > SESSIONS_MAX) {
    const oldest = sessions.keys().next().value;
    if (oldest === undefined) break;
    sessions.delete(oldest);
  }
  return fresh;
}

export interface HistorySessionState {
  /** The upstream prefix cache is provably gone — re-cutting the grid is free. */
  cold: boolean;
  /** Floor for the freeze step, in messages. 0 = no constraint. */
  minFreezeStep: number;
}

/** Neutral answer for callers without a session identity (no fingerprint yet). */
const UNKNOWN_STATE: HistorySessionState = { cold: false, minFreezeStep: 0 };

/**
 * Record a request for `sessionKey` and report what the history collapse may
 * assume about the upstream cache. Call once per transformed request, BEFORE the
 * collapse runs; it advances the session's last-seen clock.
 *
 * A session we have never seen counts as warm (see module docs) — unknown must
 * never authorize a repack.
 */
export function noteHistoryRequest(
  sessionKey: string | undefined,
  nowMs: number = Date.now(),
): HistorySessionState {
  if (!sessionKey) return UNKNOWN_STATE;
  const rec = touch(sessionKey);
  const known = rec.lastSeenMs > 0;
  const idleMs = nowMs - rec.lastSeenMs;
  const expired = known && idleMs > CACHE_TTL_SEC * 1000 + COLD_GRACE_MS;
  const cold = rec.cacheDead || expired;
  rec.lastSeenMs = nowMs;
  rec.cacheDead = false; // consumed: this request gets the repack
  return { cold, minFreezeStep: rec.freezeStep };
}

/**
 * Pin the grid this session was last rendered at. Monotonic: the floor only ever
 * rises, because a later, finer render would re-key every chunk it re-cuts.
 */
export function recordFreezeStep(
  sessionKey: string | undefined,
  step: number | undefined,
): void {
  if (!sessionKey || !step || !Number.isFinite(step) || step <= 0) return;
  const rec = touch(sessionKey);
  if (step > rec.freezeStep) rec.freezeStep = step;
}

/**
 * Mark this session's upstream cache as gone: the last request was rejected, so
 * nothing was cached and the next one may re-cut the grid for density. Call on
 * the failure paths that leave no cache entry (oversized request → opaque 500).
 */
export function markCacheDead(sessionKey: string | undefined): void {
  if (!sessionKey) return;
  touch(sessionKey).cacheDead = true;
}

/**
 * Did this response leave the upstream prefix cache unpopulated?
 *
 * A cache entry is written by a request the provider actually *accepted*. Three
 * outcomes mean it never got that far, so the frozen grid we were protecting
 * protects nothing and the next turn may re-cut for density:
 *
 *  - `413` — payload rejected outright;
 *  - `400` whose body says the prompt is too long (Anthropic's wording varies:
 *    `prompt is too long`, `prompt_too_long`, `request_too_large`);
 *  - `5xx` — includes the opaque `500` an over-cap image count produces.
 *
 * Every other 4xx (bad key, rate limit, overloaded) says nothing about the
 * cache: the prefix may well still be live, so we leave the session warm and
 * keep the cheap append-only path. Guessing "cold" there would re-cut a live
 * grid and burn the whole prefix as `cache_create` — the exact failure this
 * module exists to avoid.
 */
export function responseLeftNoCache(status: number, errorBody?: string): boolean {
  if (status === 413) return true;
  if (status >= 500) return true;
  if (status === 400 && errorBody) {
    return /prompt[\s_-]*(is\s*)?too[\s_-]*long|request[\s_-]*too[\s_-]*large|too many (images|tokens)/i
      .test(errorBody);
  }
  return false;
}

/** Test seam: drop all session state. */
export function resetSessionState(): void {
  sessions.clear();
}

/** Test/telemetry seam: inspect a session without mutating its clock. */
export function peekSessionState(
  sessionKey: string,
): { lastSeenMs: number; freezeStep: number; cacheDead: boolean } | undefined {
  const rec = sessions.get(sessionKey);
  return rec ? { ...rec } : undefined;
}

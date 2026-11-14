// The trial brief: its three JSON documents, the provider catalog probe, and the
// quality-ordered routing plan.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) carries three documents, found
// by their `schema` member, plus prose that does not matter:
//   - providers  (free-model-aggregator-01/providers/v1): the ten providers, each
//     with id, wire, auth scheme, key variable, the key the service takes, and the
//     base path joined to MAKEFUN_ENV_URL.
//   - ratings    (free-model-aggregator-01/ratings/v1): the sealed quality table:
//     every free model's rating in milli-points, context length, capabilities
//     (tools, vision, json) and free flag, plus each request's best rating.
//   - workload   (free-model-aggregator-01/workload/v1): the twenty requests in
//     order, each with api, stream flag, weight and requirements.
//
// The catalog is built by probing every provider's own catalog route in its own
// wire. The probe reports what each listing said, with the quality table's free
// flag and context length where it has a row (the table is the sealed truth).
// A provider that does not answer is left out of the catalog and gets no traffic.
//
// Routing plan: for every request, the free models meeting its requirements,
// ranked by rating. A static primary is picked per request up front — the
// highest-rated candidate whose provider still has room under that provider's
// per-trial minimum budget (the sealed table's calls_min; the real budget is
// drawn per trial between calls_min and calls_max, so staying under the minimum
// never exhausts one) — walking the requests heaviest first, so a weight-2
// request never loses its best model to a weight-1 one. At run time a refusal
// falls through the same ranking; the plan is a preference, never a promise.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (or absolute URL) the provider's routes hang off. */
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: Provider[];
  ratings: RatingRow[];
  /** Best rating per request id, from the quality table. */
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: Provider;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** One routable (provider, model) pair, with the rating that orders it. */
export interface Candidate {
  provider: Provider;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

// ---------------------------------------------------------------------------
// The sealed per-trial budget band per provider (the kit's provider table):
// a trial draws the budget between SOFT_MIN and HARD_MAX. Planning under the
// minimum never spends a provider dry; a provider past its maximum certainly is.
/** The minimum budget a trial can draw for the provider: the safe planning cap. */
export const SOFT_MIN: Readonly<Record<string, number>> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** The maximum budget a trial can draw: past this a provider is certainly spent. */
export const HARD_MAX: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

// ---------------------------------------------------------------------------
// Brief parsing

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** Index of the `}` closing the object opening at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const asBoolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function providerOf(v: unknown): Provider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // `path` first (/p/<id>), then base_path / base_url / base for a directory that
  // spells an absolute base instead.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli:
      typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`; an empty provider list and an error when it cannot be read. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && asString(v.id) !== '' && typeof v.rating_milli === 'number') {
        best.set(v.id, v.rating_milli);
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadRequest[] =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe

/** A base path joined to the env URL; an absolute base URL is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Whether a listing's pricing says every price is zero (null: it says nothing). */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context =
    asNumberOrNull(v.context_length) ?? asNumberOrNull(v.context_window) ?? asNumberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBoolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route in its own wire; a failure leaves it out. */
export async function probeProvider(envUrl: string, provider: Provider, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: readonly RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/**
 * The catalog the probes and the quality table make. Its base_url is the
 * directory's own path (/p/<id>), never joined to the env URL: the env's origin
 * differs between a live pass and its replay, and the catalog is part of the
 * state export a replay must reproduce byte for byte.
 */
export function buildCatalog(probes: readonly Probe[], ratings: readonly RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Candidates and the static plan

function capsOf(model: ProbedModel, row: RatingRow | null): { tools: boolean; vision: boolean; json: boolean; context_length: number | null } {
  return {
    tools: row?.tools ?? model.tools ?? false,
    vision: row?.vision ?? model.vision ?? false,
    json: row?.json ?? model.json ?? false,
    context_length: row?.context_length ?? model.context_length ?? null,
  };
}

/** Whether a candidate's capabilities meet a request's requirements. */
export function meetsRequirements(requires: Requirement, caps: { tools: boolean; vision: boolean; json: boolean; context_length: number | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * Every free (provider, model) pair meeting `requires`, best rating first.
 * `jsonWire` is true when the client asked for JSON mode on the OpenAI wire:
 * an Anthropic-wire provider cannot carry response_format, so it is not a
 * legal pick for such a request even when the model itself supports JSON.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: readonly RatingRow[],
  requires: Requirement,
  jsonWire: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capsOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      out.push({
        provider: probe.provider,
        model: model.id,
        rating_milli: row?.rating_milli ?? 0,
        context_length: caps.context_length,
        tools: caps.tools,
        vision: caps.vision,
        json: caps.json,
      });
    }
  }
  out.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/**
 * The static primary per request id. Requests are walked heaviest first, then
 * in manifest order; each takes the best candidate whose provider still has
 * room under its minimum budget, else the best candidate outright.
 */
export function planPrimaries(
  probes: readonly Probe[],
  ratings: readonly RatingRow[],
  manifest: readonly WorkloadRequest[],
  jsonWireOf: (req: WorkloadRequest) => boolean,
): Map<string, Candidate> {
  const planned = new Map<string, number>();
  const primaries = new Map<string, Candidate>();
  const ordered = [...manifest].sort((a, b) => b.weight - a.weight || manifest.indexOf(a) - manifest.indexOf(b));
  for (const req of ordered) {
    const candidates = candidatesFor(probes, ratings, req.requires, jsonWireOf(req));
    if (candidates.length === 0) continue;
    const pick =
      candidates.find((c) => (planned.get(c.provider.id) ?? 0) < (SOFT_MIN[c.provider.id] ?? 0)) ?? candidates[0]!;
    primaries.set(req.id, pick);
    planned.set(pick.provider.id, (planned.get(pick.provider.id) ?? 0) + 1);
  }
  return primaries;
}

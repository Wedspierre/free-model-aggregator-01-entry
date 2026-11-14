// Free model aggregator: one server, two client wires, ten free providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
// in JSON and in server-sent events, and GET /v1/models in the shape of the
// auth header it got. Every request is answered by a free model that meets its
// requirements, picked by the sealed quality table's rating, on that provider's
// own wire (openai, anthropic or gemini). A provider that refuses (402, 429,
// 5xx) costs one call and the request is answered by the next best model that
// can serve it; a provider that failed three times running is set aside until it
// answers again. Nothing here reads a clock or an unseeded random source, so
// a replay of the trial sees the same answers.
//
// Run with --list to print the catalog document and exit 0. Otherwise the
// server listens on a port it picks, registers it with the env
// (POST $MAKEFUN_ENV_URL/register), serves the client's requests, polls
// GET $MAKEFUN_ENV_URL/work, and exits 0 when the env says the workload is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  HARD_MAX,
  isRecord,
  joinBase,
  planPrimaries,
  probeProvider,
  readBrief,
  type Brief,
  type Candidate,
  type Catalog,
  type Json,
  type Probe,
  type Requirement,
  type WorkloadRequest,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicJson,
  parseAnthropicSse,
  parseGeminiJson,
  parseGeminiSse,
  parseOpenAIJson,
  parseOpenAISse,
  toAnthropicJson,
  toAnthropicSse,
  toGemini,
  toOpenAIJson,
  toOpenAISse,
  type Normalized,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The longest a retry-after pause may hold a request, in seconds. */
const MAX_RETRY_AFTER_S = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let primaries = new Map<string, Candidate>();
/** How many client API requests have arrived, in manifest order. */
let manifestIndex = 0;

/** Calls made per provider this trial, failures included: every call spends budget. */
const used = new Map<string, number>();
/** Consecutive failed calls per provider; three running is a provider set aside. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  // The static plan needs the JSON-wire flag per request; the manifest carries
  // no bodies, so a request that requires JSON on the OpenAI wire is the one
  // whose client body would carry response_format.
  primaries = planPrimaries(probes, brief.ratings, brief.manifest, (req) => req.api === 'openai' && req.requires.json === true);
}

/** The free models the catalog registers, one row per model id, best provider first. */
function servedModels(): ServedModel[] {
  const byId = new Map<string, ServedModel>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      if (!byId.has(model.id)) {
        byId.set(model.id, { id: model.id, provider: provider.id, display_name: model.id });
      }
    }
  }
  return [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Whether a provider is set aside: three failed calls running, or past its
 * maximum budget. Three, not two: a provider's failure schedule often fails
 * two calls in a row (a drawn pair like [2, 3]), and the call after the pair
 * still answers — setting it aside on two would throw away its best model for
 * the rest of the trial, as trial-1 showed when cloudflare's calls 4 and 5
 * failed and its call 6 would have served. Three running failures is a much
 * stronger sign the budget is spent or the outage has begun.
 */
function setAside(providerId: string): boolean {
  if ((consecutiveFailures.get(providerId) ?? 0) >= 3) return true;
  return (used.get(providerId) ?? 0) >= (HARD_MAX[providerId] ?? Number.MAX_SAFE_INTEGER);
}

function markSuccess(providerId: string): void {
  used.set(providerId, (used.get(providerId) ?? 0) + 1);
  consecutiveFailures.set(providerId, 0);
}

function markFailure(providerId: string): void {
  used.set(providerId, (used.get(providerId) ?? 0) + 1);
  consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
}

/**
 * The attempt order for one request: the planned primary first when it is
 * still usable, then every other usable candidate by rating, then the set-aside
 * ones by rating. A set-aside provider is only tried when nothing usable is
 * left: its budget may be spent or it may be in outage.
 */
function attemptOrder(primary: Candidate | undefined, candidates: Candidate[]): Candidate[] {
  const seen = new Set<string>();
  const key = (c: Candidate): string => `${c.provider.id}/${c.model}`;
  const usable = candidates.filter((c) => !setAside(c.provider.id));
  const aside = candidates.filter((c) => setAside(c.provider.id));
  const order: Candidate[] = [];
  if (primary !== undefined && usable.some((c) => key(c) === key(primary))) order.push(primary);
  for (const c of usable) if (!order.some((o) => key(o) === key(c))) order.push(c);
  for (const c of aside) if (!order.some((o) => key(o) === key(c))) order.push(c);
  return order.filter((c) => {
    const k = key(c);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json, req: WorkloadRequest | undefined): Requirement {
  if (req !== undefined) return req.requires;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** One candidate's upstream request, in that provider's own wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model: candidate.model } : anthropicToOpenAI(body, candidate.model);
    if (stream) {
      upstream.stream = true;
      upstream.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(upstream) };
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model } : openAIToAnthropic(body, candidate.model);
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(upstream) };
  }
  const gemini = toGemini(clientApi, body, candidate.model);
  return { url: `${base}${gemini.path}`, headers, body: JSON.stringify(gemini.body) };
}

/** The error code of a provider's refusal, in its wire's envelope. */
function errorCodeOf(wire: string, body: unknown): string {
  if (!isRecord(body)) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error === null) return typeof body.code === 'string' ? body.code : '';
  if (typeof error.code === 'string') return error.code;
  if (typeof error.type === 'string') return error.type;
  if (typeof error.status === 'string') return error.status;
  return '';
}

function retryAfterOf(res: Response, body: unknown): number {
  const header = res.headers.get('retry-after');
  if (header !== null) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds > 0) return Math.min(seconds, MAX_RETRY_AFTER_S);
  }
  if (isRecord(body) && isRecord(body.error) && typeof body.error.retry_after === 'number' && body.error.retry_after > 0) {
    return Math.min(body.error.retry_after, MAX_RETRY_AFTER_S);
  }
  return 0;
}

interface Attempt {
  ok: boolean;
  answer: Normalized | null;
  status: number;
  code: string;
  retryAfter: number;
  error: string | null;
}

/** One call to one candidate. A 200 is always kept: a second successful call would break the one-provider rule. */
async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Attempt> {
  const call = upstreamCall(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: call.body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, answer: null, status: 0, code: 'fetch_failed', retryAfter: 0, error: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      // Not JSON: the status alone says it refused.
    }
    return {
      ok: false,
      answer: null,
      status: res.status,
      code: errorCodeOf(candidate.provider.wire, parsed),
      retryAfter: retryAfterOf(res, parsed),
      error: `HTTP ${res.status}`,
    };
  }
  const stream = body.stream === true;
  let answer: Normalized | null;
  if (candidate.provider.wire === 'openai') answer = stream ? parseOpenAISse(text) : parseOpenAIJson(tryJson(text));
  else if (candidate.provider.wire === 'anthropic') answer = stream ? parseAnthropicSse(text) : parseAnthropicJson(tryJson(text));
  else answer = stream ? parseGeminiSse(text) : parseGeminiJson(tryJson(text));
  if (answer === null) {
    // A 200 that does not parse is still the one answer: keep it whole rather
    // than spending a second successful call on the same request.
    answer = { text, toolCalls: [], stopFamily: 0, usage: { input: 0, output: 0 } };
  }
  return { ok: true, answer, status: 200, code: '', retryAfter: 0, error: null };
}

function tryJson(text: string): Json {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

/** Serve one client request: the best free model that answers, with failover. */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const requires = requirementsFor(clientApi, body, entry);
  const jsonWire =
    clientApi === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(probes, brief.ratings, requires, jsonWire);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model meets this request');
  }
  const primary = entry !== undefined ? primaries.get(entry.id) : undefined;
  const order = attemptOrder(primary !== undefined && candidates.some((c) => c.provider.id === primary.provider.id && c.model === primary.model) ? primary : undefined, candidates);
  let lastFailure: Attempt | null = null;
  for (const candidate of order) {
    const result = await attempt(candidate, clientApi, body);
    if (result.ok && result.answer !== null) {
      markSuccess(candidate.provider.id);
      const model = candidate.model;
      if (body.stream === true) {
        return sendSse(res, clientApi === 'openai' ? toOpenAISse(result.answer, model) : toAnthropicSse(result.answer, model));
      }
      return sendJson(res, 200, clientApi === 'openai' ? toOpenAIJson(result.answer, model) : toAnthropicJson(result.answer, model));
    }
    markFailure(candidate.provider.id);
    lastFailure = result;
    log(`refused: ${candidate.provider.id}/${candidate.model} HTTP ${result.status} ${result.code}`);
    if (result.retryAfter > 0) await sleep(result.retryAfter * 1000);
  }
  const detail = lastFailure === null ? 'no candidate answered' : `the last refusal was HTTP ${lastFailure.status} ${lastFailure.code}`;
  return sendError(res, clientApi, 503, 'no_free_model', `no free model answered this request (${detail})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} probed providers`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

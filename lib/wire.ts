// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), in JSON or in
// server-sent events. Each request is answered by one of the ten providers, on
// that provider's own wire (openai, anthropic or gemini), and the answer is
// returned in the wire the client used.
//
// The provider echoes, in its answer text, what it parsed from the request it
// got: system prompt, user text, images, tool names, max_tokens, stop
// sequences, JSON mode. The client compares that echo with what it sent, so a
// translation must carry every one of those fields across, untouched in value,
// whatever wire it crosses. Nothing else in a body is echoed, but the faithful
// fields are passed through where the wire carries them.
//
// Every function here is a pure function of its arguments: no clock, no
// randomness. Streaming answers are buffered, parsed into one normalized
// answer, and re-emitted in the client's event order, so every direction
// produces a well-formed stream: the provider's text whole, its usage counts
// as that wire reports them, its stop reason in the same family.

import { isRecord, type Json } from './brief.ts';

/** One tool call in a normalized answer; `args` is the parsed arguments object. */
export interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

/** Stop-reason families, shared by the three wires. */
export type StopFamily = 0 | 1 | 2 | 3; // 0 stop, 1 length, 2 tool_use, 3 refusal

/** One provider answer, in wire-neutral form. */
export interface Normalized {
  text: string;
  toolCalls: ToolCall[];
  stopFamily: StopFamily;
  usage: { input: number; output: number };
}

/** Fixed epoch second for every listed model's `created`: nothing reads a clock. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Fixed ids for answers this server builds itself. */
const OPENAI_ANSWER_ID = 'chatcmpl-agg';
const ANTHROPIC_ANSWER_ID = 'msg_agg';

// ---------------------------------------------------------------------------
// small helpers

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function openAIFamily(finish: unknown): StopFamily {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

function anthropicFamily(stop: unknown): StopFamily {
  if (stop === 'max_tokens') return 1;
  if (stop === 'tool_use') return 2;
  if (stop === 'refusal') return 3;
  return 0;
}

function geminiFamily(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY') return 3;
  return 0;
}

const openAIFinishOf = (family: StopFamily, toolCalls: boolean): string =>
  toolCalls ? 'tool_calls' : family === 1 ? 'length' : family === 3 ? 'content_filter' : 'stop';
const anthropicStopOf = (family: StopFamily, toolCalls: boolean): string =>
  toolCalls ? 'tool_use' : family === 1 ? 'max_tokens' : family === 3 ? 'refusal' : 'end_turn';

// ---------------------------------------------------------------------------
// model lists

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models on an OpenAI-shaped request: the served free model ids. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// client -> upstream: OpenAI <-> Anthropic

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: system becomes a system message, image blocks become image_url data
 * URLs, tool_use blocks become tool_calls, tool_result blocks become tool
 * messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) {
          blocks.push({
            type: 'image',
            source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[2] },
          });
        } else {
          blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (max !== undefined) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// client -> upstream: Gemini

/** The last user text of a client body (OpenAI or Anthropic): the echo's user text. */
function lastUserText(clientApi: 'openai' | 'anthropic', body: Json): string {
  const source = Array.isArray(body.messages) ? body.messages : [];
  let text = '';
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    const t = textOf(raw.content);
    if (t !== '') text = t;
  }
  return text;
}

/** An image part as a Gemini part: inlineData for data URLs, fileData otherwise. */
function geminiImagePart(url: string): Json | null {
  if (url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[2] } };
  }
  return { fileData: { fileUri: url } };
}

/**
 * Either client wire as a Gemini generateContent body naming `model` in the
 * URL (the body itself carries no model). System becomes systemInstruction,
 * user/assistant turns become contents, tool definitions become
 * functionDeclarations, and maxOutputTokens / stopSequences / responseMimeType
 * carry the cap, the stops and JSON mode. A trailing tool-result turn would
 * leave the provider's parsed user text empty, so the last non-model content
 * always carries the last user text.
 */
export function toGemini(clientApi: 'openai' | 'anthropic', body: Json, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const tools: Json[] = [];
  let maxTokens: number | undefined;
  let stops: string[] = [];
  let jsonMode = false;
  let temperature: number | undefined;
  let topP: number | undefined;

  const addFunctionDeclarations = (decls: Json[]): void => {
    if (decls.length > 0) tools.push({ functionDeclarations: decls });
  };

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemParts.push(t);
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({
          role: 'user',
          parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
        });
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const t = textOf(raw.content);
        if (t !== '') parts.push({ text: t });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({
            functionCall: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              args: argumentsOf(call.function.arguments),
            },
          });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const image = geminiImagePart(part.image_url.url);
            if (image !== null) parts.push(image);
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    else if (typeof body.stop === 'string' && body.stop !== '') stops = [body.stop];
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) jsonMode = true;
    if (Array.isArray(body.tools)) {
      addFunctionDeclarations(
        body.tools.flatMap((raw) => {
          if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
          return [
            {
              name: raw.function.name,
              description: typeof raw.function.description === 'string' ? raw.function.description : '',
              parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
            },
          ];
        }),
      );
    }
  } else {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        if (typeof content === 'string') {
          if (content !== '') parts.push({ text: content });
        } else if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              parts.push({
                functionCall: {
                  name: typeof block.name === 'string' ? block.name : '',
                  args: isRecord(block.input) ? block.input : {},
                },
              });
            }
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            const src = block.source;
            if (src.type === 'base64' && typeof src.data === 'string') {
              parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
            } else if (src.type === 'url' && typeof src.url === 'string') {
              const image = geminiImagePart(src.url);
              if (image !== null) parts.push(image);
            }
          } else if (block.type === 'tool_result') {
            parts.push({
              functionResponse: {
                name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
                response: { content: toolResultText(block.content) },
              },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      addFunctionDeclarations(
        body.tools.flatMap((raw) => {
          if (!isRecord(raw) || typeof raw.name !== 'string') return [];
          return [
            {
              name: raw.name,
              description: typeof raw.description === 'string' ? raw.description : '',
              parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
            },
          ];
        }),
      );
    }
  }

  if (typeof body.temperature === 'number') temperature = body.temperature;
  if (typeof body.top_p === 'number') topP = body.top_p;

  // The provider parses the user text of the last non-model content; a turn
  // of tool results alone would read as no user text at all. Carry the last
  // user text there when the turn has no text part of its own.
  for (let i = contents.length - 1; i >= 0; i--) {
    const content = contents[i]!;
    if (content.role === 'model') continue;
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const hasText = parts.some((p) => isRecord(p) && typeof p.text === 'string');
    if (!hasText) {
      const text = lastUserText(clientApi, body);
      if (text !== '') parts.push({ text });
    }
    break;
  }

  const generationConfig: Json = {};
  if (maxTokens !== undefined) generationConfig.maxOutputTokens = maxTokens;
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  if (temperature !== undefined) generationConfig.temperature = temperature;
  if (topP !== undefined) generationConfig.topP = topP;

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  const stream = body.stream === true;
  return {
    path: `/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}${stream ? '?alt=sse' : ''}`,
    body: out,
  };
}

// ---------------------------------------------------------------------------
// upstream answers -> normalized

/** Split raw SSE text into frames: each frame's event name and data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonPayload(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function textFromContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** An OpenAI JSON completion as a normalized answer. */
export function parseOpenAIJson(body: Json): Normalized | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : 'call_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args: argumentsOf(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textFromContent(message.content),
    toolCalls,
    stopFamily: openAIFamily(choice.finish_reason),
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
  };
}

/** An OpenAI SSE stream as a normalized answer. */
export function parseOpenAISse(text: string): Normalized | null {
  let out = '';
  let family: StopFamily = 0;
  let input = 0;
  let output = 0;
  let sawAny = false;
  const toolParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJsonPayload(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      const slot = toolParts.get(index) ?? { id: '', name: '', args: '' };
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
      if (typeof fn.arguments === 'string') slot.args += fn.arguments;
      toolParts.set(index, slot);
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) family = openAIFamily(choice.finish_reason);
  }
  if (!sawAny) return null;
  const toolCalls: ToolCall[] = [...toolParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id !== '' ? slot.id : 'call_unknown', name: slot.name, args: argumentsOf(slot.args) }));
  return { text: out, toolCalls, stopFamily: family, usage: { input, output } };
}

/** An Anthropic JSON message as a normalized answer. */
export function parseAnthropicJson(body: Json): Normalized | null {
  if (!Array.isArray(body.content)) return null;
  const toolCalls: ToolCall[] = [];
  let out = '';
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') out += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        name: typeof block.name === 'string' ? block.name : '',
        args: isRecord(block.input) ? block.input : {},
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: out,
    toolCalls,
    stopFamily: anthropicFamily(body.stop_reason),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
  };
}

/** An Anthropic SSE stream as a normalized answer. */
export function parseAnthropicSse(text: string): Normalized | null {
  let out = '';
  let family: StopFamily = 0;
  let input = 0;
  let output = 0;
  let sawAny = false;
  const toolParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonPayload(frame.data);
    if (data === null) continue;
    sawAny = true;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          input = numberOrZero(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : 0;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolParts.set(index, {
            id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
            name: typeof block.name === 'string' ? block.name : '',
            args: '',
          });
        }
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : 0;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          const slot = toolParts.get(index);
          if (slot !== undefined) slot.args += delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') family = anthropicFamily(delta.stop_reason);
        if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  if (!sawAny) return null;
  const toolCalls: ToolCall[] = [...toolParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id, name: slot.name, args: argumentsOf(slot.args) }));
  return { text: out, toolCalls, stopFamily: family, usage: { input, output } };
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini generateContent answer as a normalized answer. */
export function parseGeminiJson(body: Json): Normalized | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiPartsText(content.parts),
    toolCalls: [],
    stopFamily: geminiFamily(candidate.finishReason),
    usage: { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** A Gemini streamGenerateContent stream as a normalized answer. */
export function parseGeminiSse(text: string): Normalized | null {
  let out = '';
  let family: StopFamily = 0;
  let input = 0;
  let output = 0;
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const chunk = parseJsonPayload(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      out += geminiPartsText(content.parts);
      if (typeof candidate.finishReason === 'string') family = geminiFamily(candidate.finishReason);
    }
    if (isRecord(chunk.usageMetadata)) {
      input = numberOrZero(chunk.usageMetadata.promptTokenCount);
      output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: out, toolCalls: [], stopFamily: family, usage: { input, output } };
}

// ---------------------------------------------------------------------------
// normalized -> client

/** One server-sent event frame. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalized answer as an OpenAI Chat Completions object. */
export function toOpenAIJson(norm: Normalized, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: norm.text,
          ...(norm.toolCalls.length > 0
            ? {
                tool_calls: norm.toolCalls.map((c) => ({
                  id: c.id,
                  type: 'function',
                  function: { name: c.name, arguments: JSON.stringify(c.args) },
                })),
              }
            : {}),
        },
        finish_reason: openAIFinishOf(norm.stopFamily, norm.toolCalls.length > 0),
      },
    ],
    usage: {
      prompt_tokens: norm.usage.input,
      completion_tokens: norm.usage.output,
      total_tokens: norm.usage.input + norm.usage.output,
    },
  };
}

/** A normalized answer as an OpenAI SSE stream, ending with usage and [DONE]. */
export function toOpenAISse(norm: Normalized, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(norm.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  norm.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null),
      ),
    );
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, function: { arguments: JSON.stringify(call.args) } }] }, null),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, openAIFinishOf(norm.stopFamily, norm.toolCalls.length > 0))));
  frames.push(
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: norm.usage.input,
        completion_tokens: norm.usage.output,
        total_tokens: norm.usage.input + norm.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages object. */
export function toAnthropicJson(norm: Normalized, model: string): Json {
  const content: Json[] = [];
  if (norm.text !== '' || norm.toolCalls.length === 0) content.push({ type: 'text', text: norm.text });
  for (const call of norm.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
  }
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(norm.stopFamily, norm.toolCalls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: norm.usage.input, output_tokens: norm.usage.output },
  };
}

/** A normalized answer as an Anthropic SSE stream, in event order. */
export function toAnthropicSse(norm: Normalized, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: norm.usage.input, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitText(norm.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of norm.toolCalls) {
    const toolIndex = index;
    index += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: toolIndex,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: toolIndex,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args) },
      }),
    );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: toolIndex }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(norm.stopFamily, norm.toolCalls.length > 0), stop_sequence: null },
      usage: { output_tokens: norm.usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

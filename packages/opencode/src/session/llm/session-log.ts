import { randomUUID } from "node:crypto"
import { appendFileSync, mkdirSync } from "node:fs"
import { dirname, extname, join } from "node:path"
import type { LLMEvent } from "@opencode-ai/llm"
import type { ModelMessage } from "ai"
import type { Agent } from "@/agent/agent"
import type { Provider } from "@/provider/provider"

type Meta = {
  readonly sessionID: string
  readonly parentSessionID?: string
  readonly agent: Agent.Info
  readonly provider: Provider.Info
  readonly model: Provider.Model
}

type Usage = {
  readonly prompt_tokens?: number
  readonly completion_tokens?: number
  readonly total_tokens?: number
  readonly reasoning_tokens?: number
  readonly cache_read_input_tokens?: number
  readonly cache_write_input_tokens?: number
}

type ToolCall = {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

type State = {
  text: string
  reasoning: string
  toolCalls: ToolCall[]
  finishReason?: string
  usage?: Usage
  error?: unknown
}

function enabled() {
  const value = process.env.OPENCODE_LLM_LOG
  return value === "1" || value === "true" || value === "yes"
}

function outputPath() {
  const target = process.env.OPENCODE_LLM_LOG_PATH ?? process.env.OPENCODE_LLM_LOG_DIR
  if (!target) return
  if (extname(target) === ".jsonl") return target
  return join(target, "llm-calls.jsonl")
}

function eventsPath() {
  const target = process.env.OPENCODE_LLM_EVENT_LOG_PATH
  if (target) return target
  const path = outputPath()
  if (!path) return
  return join(dirname(path), "llm-events.jsonl")
}

function appendJsonl(path: string, row: unknown) {
  mkdirSync(dirname(path), { recursive: true })
  appendFileSync(path, JSON.stringify(compact(row as object)) + "\n")
}

function usage(input: Extract<LLMEvent, { type: "finish" }>["usage"] | undefined): Usage | undefined {
  if (!input) return
  return {
    prompt_tokens: input.inputTokens,
    completion_tokens: input.outputTokens,
    total_tokens: input.totalTokens,
    reasoning_tokens: input.reasoningTokens,
    cache_read_input_tokens: input.cacheReadInputTokens,
    cache_write_input_tokens: input.cacheWriteInputTokens,
  }
}

function compact<T extends object>(value: T): T {
  return JSON.parse(JSON.stringify(value))
}

function normalizeMessages(messages: ModelMessage[]) {
  return messages.map((message) => {
    if (message.role === "system") return { role: "system", content: message.content }
    if (message.role === "user") return { role: "user", content: message.content }
    if (message.role === "assistant") return { role: "assistant", content: message.content }
    return message
  })
}

function normalizeToolInput(input: unknown) {
  if (typeof input === "string") return input
  return JSON.stringify(input ?? {})
}

export function create(meta: Meta, request: { messages: ModelMessage[] }) {
  if (!enabled()) return
  const path = outputPath()
  if (!path) return

  const startedAt = Date.now()
  const callID = randomUUID()
  const state: State = {
    text: "",
    reasoning: "",
    toolCalls: [],
  }

  const record = {
    call_id: callID,
    session_id: meta.sessionID,
    parent_session_id: meta.parentSessionID,
    agent: meta.agent.name,
    provider: meta.model.providerID,
    model: meta.model.id,
    provider_name: meta.provider.name,
    request: {
      model: `${meta.model.providerID}/${meta.model.id}`,
      messages: normalizeMessages(request.messages),
    },
  }
  const eventPath = eventsPath()
  if (eventPath) {
    appendJsonl(eventPath, {
      time: new Date(startedAt).toISOString(),
      event: "request_started",
      call_id: callID,
      session_id: record.session_id,
      parent_session_id: record.parent_session_id,
      agent: record.agent,
      provider: record.provider,
      model: record.model,
      request: {
        model: record.request.model,
        message_count: record.request.messages.length,
      },
    })
  }

  return {
    event(event: LLMEvent) {
      if (event.type === "text-delta") {
        state.text += event.text
        return
      }
      if (event.type === "reasoning-delta") {
        state.reasoning += event.text
        return
      }
      if (event.type === "tool-call") {
        state.toolCalls.push({
          id: event.id,
          type: "function",
          function: {
            name: event.name,
            arguments: normalizeToolInput(event.input),
          },
        })
        return
      }
      if (event.type === "finish") {
        state.finishReason = event.reason
        state.usage = usage(event.usage)
      }
    },
    error(error: unknown) {
      state.error = error
    },
    flush() {
      const response = {
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: state.text,
              ...(state.toolCalls.length > 0 ? { tool_calls: state.toolCalls } : {}),
            },
            finish_reason: state.finishReason,
          },
        ],
        ...(state.usage ? { usage: state.usage } : {}),
      }
      const row = compact({
        time: new Date(startedAt).toISOString(),
        ...record,
        response,
        ...(state.reasoning ? { reasoning: state.reasoning } : {}),
        ...(state.error ? { error: String(state.error) } : {}),
      })
      appendJsonl(path, row)
      if (eventPath) {
        appendJsonl(eventPath, {
          time: new Date().toISOString(),
          event: "request_finished",
          call_id: callID,
          session_id: record.session_id,
          parent_session_id: record.parent_session_id,
          agent: record.agent,
          provider: record.provider,
          model: record.model,
          finish_reason: state.finishReason,
          usage: state.usage,
          error: state.error ? String(state.error) : undefined,
        })
      }
    },
  }
}

export * as LLMSessionLog from "./session-log"

import type OpenAI from 'openai'
import { createInterface } from 'node:readline/promises'
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionSystemMessageParam,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
  ChatCompletionUserMessageParam,
} from 'openai/resources/chat/completions'

export type PromptRole = 'system'
export type PromptMessage = ChatCompletionSystemMessageParam
export type AgentMessage = Exclude<
  ChatCompletionMessageParam,
  { role: 'function' }
>

export interface MemoryState {
  value: string
}

interface SummarizeMemoryOptions {
  client: OpenAI
  summaryModel: string
  messages: AgentMessage[]
  existingMemory: string
  maxMemoryTokens: number
  instructionRole: PromptRole
}

interface CompactMessagesOptions {
  messages: AgentMessage[]
  memory: MemoryState
  tools: ChatCompletionTool[]
  maxContextMessages: number
  summaryTriggerTokens: number
  summarize: (
    messages: AgentMessage[],
    existingMemory: string,
  ) => Promise<string>
}

const CJK_CHAR_PATTERN =
  /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu
const SUMMARY_INSTRUCTION =
  '你负责压缩对话上下文。请把历史消息总结成一段可继续推理的 memory，保留：用户目标、关键事实、已经调用过的工具与结果、已做出的决定、仍未完成的任务。不要写寒暄，不要重复无关细节。'

export function readNumericEnv(name: string, fallback: number): number {
  const rawValue = process.env[name]
  if (!rawValue) {
    return fallback
  }

  const parsed = Number(rawValue)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function inheritProcessEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const [, value] = entry
      return value !== undefined
    }),
  )
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return String(error)
}

export async function readPromptFromCli(
  question: string,
  promptArgs: string[] = process.argv.slice(2),
): Promise<string> {
  const argPrompt = promptArgs.join(' ').trim()
  if (argPrompt) {
    return argPrompt
  }

  if (!process.stdin.isTTY) {
    let pipedPrompt = ''
    for await (const chunk of process.stdin) {
      pipedPrompt += typeof chunk === 'string' ? chunk : chunk.toString('utf8')
    }

    const trimmedPrompt = pipedPrompt.trim()
    if (trimmedPrompt) {
      return trimmedPrompt
    }
  }

  const readline = createInterface({
    input: process.stdin,
    output: process.stdout,
  })

  try {
    const prompt = (await readline.question(question)).trim()
    if (!prompt) {
      throw new Error('prompt 不能为空')
    }

    return prompt
  } finally {
    readline.close()
  }
}

export function parseObjectArguments(
  rawArguments: string,
): Record<string, unknown> {
  const parsed = rawArguments ? (JSON.parse(rawArguments) as unknown) : {}
  if (!isRecord(parsed)) {
    throw new Error('tool arguments 必须是 JSON 对象')
  }

  return parsed
}

export function normalizeMessageContent(
  content: string | Array<{ text?: string; type?: string }> | null | undefined,
): string {
  if (typeof content === 'string') {
    return content
  }

  if (!content) {
    return ''
  }

  return content
    .map((part) =>
      typeof part.text === 'string' ? part.text : JSON.stringify(part),
    )
    .join('\n')
}

export function estimateTokensFromText(text: string): number {
  if (!text) {
    return 0
  }

  const cjkMatches = text.match(CJK_CHAR_PATTERN) ?? []
  const nonCjk = text.replace(CJK_CHAR_PATTERN, '')
  const nonWhitespaceChars = nonCjk.replace(/\s+/g, '').length

  return cjkMatches.length + Math.ceil(nonWhitespaceChars / 4)
}

export function clipText(text: string, maxTokens: number): string {
  if (estimateTokensFromText(text) <= maxTokens) {
    return text
  }

  const suffix = '\n...<truncated>'
  const allowedTokens = Math.max(1, maxTokens - estimateTokensFromText(suffix))
  let left = 0
  let right = text.length
  let best = ''

  // 用二分截断可在超长文本下保持较高性能。
  while (left <= right) {
    const mid = Math.floor((left + right) / 2)
    const candidate = text.slice(0, mid)

    if (estimateTokensFromText(candidate) <= allowedTokens) {
      best = candidate
      left = mid + 1
    } else {
      right = mid - 1
    }
  }

  return `${best}${suffix}`
}

export function createPromptMessage(
  role: PromptRole,
  content: string,
): PromptMessage {
  return {
    role: 'system',
    content,
  }
}

export function createUserMessage(
  content: string,
): ChatCompletionUserMessageParam {
  return {
    role: 'user',
    content,
  }
}

export function createAssistantToolCallMessage(
  content: string,
  toolCalls: ChatCompletionMessageFunctionToolCall[],
): ChatCompletionAssistantMessageParam {
  return {
    role: 'assistant',
    content,
    tool_calls: toolCalls,
  }
}

export function createToolMessage(
  toolCallId: string,
  content: string,
): ChatCompletionToolMessageParam {
  return {
    role: 'tool',
    tool_call_id: toolCallId,
    content,
  }
}

export function getFunctionToolCalls(
  toolCalls: ChatCompletionMessageToolCall[] | undefined,
): ChatCompletionMessageFunctionToolCall[] {
  return (toolCalls ?? []).map((toolCall) => {
    if (toolCall.type !== 'function') {
      throw new Error(`Unsupported tool call type: ${toolCall.type}`)
    }

    return toolCall
  })
}

function isPromptMessage(message: AgentMessage): message is PromptMessage {
  return message.role === 'system'
}

function messageToMemoryLine(message: AgentMessage): string {
  if (message.role === 'user') {
    return `User: ${clipText(normalizeMessageContent(message.content), 240)}`
  }

  if (
    message.role === 'assistant' &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    const names = getFunctionToolCalls(message.tool_calls)
      .map((toolCall) => toolCall.function.name)
      .join(', ')
    return `Assistant called tools: ${names}`
  }

  if (message.role === 'assistant') {
    return `Assistant: ${clipText(normalizeMessageContent(message.content), 240)}`
  }

  if (message.role === 'tool') {
    return `Tool result: ${clipText(normalizeMessageContent(message.content), 240)}`
  }

  return `Instruction: ${clipText(normalizeMessageContent(message.content), 240)}`
}

function formatMessageForSummary(message: AgentMessage): string {
  if (message.role === 'user') {
    return `[user]\n${clipText(normalizeMessageContent(message.content), 1200)}`
  }

  if (
    message.role === 'assistant' &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    const toolLines = getFunctionToolCalls(message.tool_calls)
      .map(
        (toolCall) =>
          `- ${toolCall.function.name}(${clipText(toolCall.function.arguments || '{}', 400)})`,
      )
      .join('\n')

    return `[assistant tool calls]\n${toolLines}`
  }

  if (message.role === 'assistant') {
    return `[assistant]\n${clipText(normalizeMessageContent(message.content), 1200)}`
  }

  if (message.role === 'tool') {
    return `[tool result]\n${clipText(normalizeMessageContent(message.content), 1200)}`
  }

  return `[instruction]\n${clipText(normalizeMessageContent(message.content), 1200)}`
}

function buildFallbackMemory(
  existingMemory: string,
  archivedMessages: AgentMessage[],
): string {
  const lines = archivedMessages.map(messageToMemoryLine).filter(Boolean)
  const merged = [existingMemory, ...lines].filter(Boolean).join('\n')
  return merged
}

export function buildMessagesForModel(
  messages: AgentMessage[],
  memory: string,
): AgentMessage[] {
  const baseMessages = [...messages]
  if (!memory) {
    return baseMessages
  }

  const firstMessage = baseMessages[0]
  if (!firstMessage || !isPromptMessage(firstMessage)) {
    throw new Error('对话的第一条消息必须是 system 提示词')
  }

  // 以额外 system 消息注入 memory，保证原始顶层提示词不被覆盖。
  return [
    firstMessage,
    createPromptMessage(
      firstMessage.role,
      `这是压缩后的历史上下文，只保留关键信息：\n${memory}`,
    ),
    ...baseMessages.slice(1),
  ]
}

function estimateContextTokens(
  messages: AgentMessage[],
  memory: string,
  tools: ChatCompletionTool[],
): number {
  return (
    estimateTokensFromText(
      JSON.stringify(buildMessagesForModel(messages, memory)),
    ) + estimateTokensFromText(JSON.stringify(tools))
  )
}

export async function summarizeMemory(
  options: SummarizeMemoryOptions,
): Promise<string> {
  const {
    client,
    summaryModel,
    messages,
    existingMemory,
    maxMemoryTokens,
    instructionRole,
  } = options

  const fallback = clipText(
    buildFallbackMemory(existingMemory, messages),
    maxMemoryTokens,
  )
  const historyText = messages
    .map(formatMessageForSummary)
    .filter(Boolean)
    .join('\n\n')

  try {
    const response = await client.chat.completions.create({
      model: summaryModel,
      messages: [
        createPromptMessage(instructionRole, SUMMARY_INSTRUCTION),
        createUserMessage(
          `已有 memory：\n${existingMemory || '（无）'}\n\n请基于下面的历史消息，生成新的 memory。\n\n历史消息：\n${historyText}`,
        ),
      ],
    })

    const summary = response.choices[0]?.message?.content?.trim()
    if (!summary) {
      return fallback
    }

    return clipText(summary, maxMemoryTokens)
  } catch (error) {
    console.error(`[context] summary failed: ${toErrorMessage(error)}`)
    return fallback
  }
}

export async function compactMessages(
  options: CompactMessagesOptions,
): Promise<void> {
  const {
    messages,
    memory,
    tools,
    maxContextMessages,
    summaryTriggerTokens,
    summarize,
  } = options

  if (
    estimateContextTokens(messages, memory.value, tools) < summaryTriggerTokens
  ) {
    return
  }

  if (messages.length <= maxContextMessages + 1) {
    return
  }

  // 永远保留首条 system 提示词和最近 N 条消息，只压缩中间历史段。
  const archived = messages.slice(1, messages.length - maxContextMessages)
  if (archived.length === 0) {
    return
  }

  memory.value = await summarize(archived, memory.value)

  const recentMessages = messages.slice(-maxContextMessages)
  messages.splice(1, messages.length - 1, ...recentMessages)

  console.log(
    `[context] summarized ${archived.length} messages into memory, approx tokens=${estimateContextTokens(
      messages,
      memory.value,
      tools,
    )}`,
  )
}

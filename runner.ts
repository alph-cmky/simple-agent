import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
} from 'openai/resources/chat/completions'
import {
  buildMessagesForModel,
  clipText,
  compactMessages,
  createAssistantToolCallMessage,
  createPromptMessage,
  createToolMessage,
  createUserMessage,
  getFunctionToolCalls,
  normalizeMessageContent,
  summarizeMemory,
  toErrorMessage,
  type AgentMessage,
  type MemoryState,
} from './shared.js'
import {
  MAX_CONTEXT_MESSAGES,
  MAX_MEMORY_TOKENS,
  MAX_TOOL_RESULT_TOKENS,
  MODEL,
  RATE_LIMIT_RETRY_DELAYS_MS,
  SUMMARY_MODEL,
  SUMMARY_TRIGGER_TOKENS,
  SYSTEM_PROMPT,
  client,
} from './config.js'
import type { ToolProvider } from './types.js'

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function isRateLimitError(error: unknown): boolean {
  const message = toErrorMessage(error).toLowerCase()
  return (
    message.includes('429') ||
    message.includes('rate limit') ||
    message.includes('rpm')
  )
}

async function createChatCompletionWithRetry(
  payload: ChatCompletionCreateParamsNonStreaming,
): Promise<ChatCompletion> {
  let lastError: unknown

  for (
    let attempt = 0;
    attempt <= RATE_LIMIT_RETRY_DELAYS_MS.length;
    attempt += 1
  ) {
    try {
      return await client.chat.completions.create(payload)
    } catch (error) {
      lastError = error
      if (
        !isRateLimitError(error) ||
        attempt >= RATE_LIMIT_RETRY_DELAYS_MS.length
      ) {
        break
      }

      const waitMs =
        RATE_LIMIT_RETRY_DELAYS_MS[
          Math.min(attempt, RATE_LIMIT_RETRY_DELAYS_MS.length - 1)
        ] || 1000
      console.error(`[model] rate limited, retrying in ${waitMs}ms...`)
      await sleep(waitMs)
    }
  }

  throw lastError
}

export async function runAgent(
  userPrompt: string,
  provider: ToolProvider,
): Promise<void> {
  const memory: MemoryState = { value: '' }
  const messages: AgentMessage[] = [
    createPromptMessage(provider.instructionRole, SYSTEM_PROMPT),
    createUserMessage(userPrompt),
  ]

  try {
    for (let step = 1; step <= provider.maxSteps; step += 1) {
      console.log(`\n[loop ${step}] calling model...`)
      await compactMessages({
        messages,
        memory,
        tools: provider.tools,
        maxContextMessages: MAX_CONTEXT_MESSAGES,
        summaryTriggerTokens: SUMMARY_TRIGGER_TOKENS,
        summarize: (archivedMessages, existingMemory) =>
          summarizeMemory({
            client,
            summaryModel: SUMMARY_MODEL,
            messages: archivedMessages,
            existingMemory,
            maxMemoryTokens: MAX_MEMORY_TOKENS,
            instructionRole: provider.instructionRole,
          }),
      })

      const response = await createChatCompletionWithRetry({
        model: MODEL,
        parallel_tool_calls: false,
        tools: provider.tools,
        messages: buildMessagesForModel(messages, memory.value),
      })

      const message = response.choices?.[0]?.message
      if (!message) {
        throw new Error(
          `模型没有返回消息: ${clipText(JSON.stringify(response), 400)}`,
        )
      }

      const toolCalls = getFunctionToolCalls(message.tool_calls)
      const assistantContent =
        normalizeMessageContent(message.content) || message.refusal || ''

      if (toolCalls.length === 0) {
        console.log('\n[final answer]')
        console.log(assistantContent)
        return
      }

      messages.push(createAssistantToolCallMessage(assistantContent, toolCalls))

      for (const toolCall of toolCalls) {
        console.log(
          `\n[${provider.toolLogLabel}] ${toolCall.function.name}(${toolCall.function.arguments})`,
        )

        let output: string
        try {
          output = await provider.executeTool(
            toolCall.function.name,
            toolCall.function.arguments,
          )
        } catch (error) {
          output = `Tool error: ${toErrorMessage(error)}`
        }

        const preview =
          output.length > provider.previewLimit
            ? `${output.slice(0, provider.previewLimit)}\n...<truncated>`
            : output

        console.log(`[${provider.resultLogLabel}]\n${preview}`)
        messages.push(
          createToolMessage(
            toolCall.id,
            clipText(output, MAX_TOOL_RESULT_TOKENS),
          ),
        )
      }
    }
  } finally {
    await provider.close()
  }

  throw new Error(`超过最大循环次数: ${provider.maxSteps}`)
}

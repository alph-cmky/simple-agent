import 'dotenv/config'
import fs from 'node:fs/promises'
import path from 'node:path'
import OpenAI from 'openai'

const API_KEY = process.env.STEP_API_KEY || process.env.OPENAI_API_KEY
const BASE_URL =
  process.env.STEP_BASE_URL ||
  process.env.OPENAI_BASE_URL ||
  'https://api.stepfun.ai/step_plan/v1'
const MODEL =
  process.env.STEP_MODEL || process.env.OPENAI_MODEL || 'step-3.5-flash'
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || MODEL
const ROOT_DIR = process.cwd()
const MAX_STEPS = 100
const MAX_CONTEXT_MESSAGES = Number(process.env.MAX_CONTEXT_MESSAGES || 12)
const MAX_MEMORY_TOKENS = Number(
  process.env.MAX_MEMORY_TOKENS ||
    Math.ceil(Number(process.env.MAX_MEMORY_CHARS || 4000) / 3),
)
const MAX_TOOL_RESULT_TOKENS = Number(
  process.env.MAX_TOOL_RESULT_TOKENS ||
    Math.ceil(Number(process.env.MAX_TOOL_RESULT_CHARS || 4000) / 3),
)
const SUMMARY_TRIGGER_TOKENS = Number(
  process.env.SUMMARY_TRIGGER_TOKENS ||
    Math.ceil(Number(process.env.SUMMARY_TRIGGER_CHARS || 12000) / 3),
)

if (!API_KEY) {
  console.error('Missing STEP_API_KEY or OPENAI_API_KEY')
  process.exit(1)
}

const client = new OpenAI({
  apiKey: API_KEY,
  baseURL: BASE_URL,
})

function resolveInsideRoot(target = '.') {
  const fullPath = path.resolve(ROOT_DIR, target)
  const relativePath = path.relative(ROOT_DIR, fullPath)

  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new Error('只能访问当前工作目录里的文件')
  }

  return fullPath
}

async function listFiles({ dir = '.' } = {}) {
  const fullPath = resolveInsideRoot(dir)
  const entries = await fs.readdir(fullPath, { withFileTypes: true })

  return entries.map((entry) => ({
    name: entry.name,
    type: entry.isDirectory() ? 'dir' : 'file',
  }))
}

async function readFile({ file_path } = {}) {
  if (!file_path) {
    throw new Error('file_path 不能为空')
  }

  const fullPath = resolveInsideRoot(file_path)
  const content = await fs.readFile(fullPath, 'utf8')

  if (content.length > 12000) {
    return `${content.slice(0, 12000)}\n...<truncated>`
  }

  return content
}

const tools = [
  {
    type: 'function',
    function: {
      name: 'list_files',
      description: '列出当前工作目录中的文件和文件夹',
      parameters: {
        type: 'object',
        properties: {
          dir: {
            type: 'string',
            description: '相对路径，默认是 .',
          },
        },
        additionalProperties: false,
      },
      strict: true,
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: '读取当前工作目录中的文本文件',
      parameters: {
        type: 'object',
        properties: {
          file_path: {
            type: 'string',
            description: '要读取的相对文件路径',
          },
        },
        required: ['file_path'],
        additionalProperties: false,
      },
      strict: true,
    },
  },
]

const toolHandlers = {
  list_files: listFiles,
  read_file: readFile,
}

function estimateTokensFromText(text) {
  if (!text) {
    return 0
  }

  const cjkMatches = text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || []
  const nonCjk = text.replace(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu, '')
  const nonWhitespaceChars = nonCjk.replace(/\s+/g, '').length

  return cjkMatches.length + Math.ceil(nonWhitespaceChars / 4)
}

function clipText(text, maxTokens) {
  if (estimateTokensFromText(text) <= maxTokens) {
    return text
  }

  const suffix = '\n...<truncated>'
  const allowedTokens = Math.max(1, maxTokens - estimateTokensFromText(suffix))
  let left = 0
  let right = text.length
  let best = ''

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

function messageToMemoryLine(message) {
  if (message.role === 'user') {
    return `User: ${clipText(String(message.content || ''), 240)}`
  }

  if (message.role === 'assistant' && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    const names = message.tool_calls.map((toolCall) => toolCall.function.name).join(', ')
    return `Assistant called tools: ${names}`
  }

  if (message.role === 'assistant') {
    return `Assistant: ${clipText(String(message.content || ''), 240)}`
  }

  if (message.role === 'tool') {
    return `Tool result: ${clipText(String(message.content || ''), 240)}`
  }

  return ''
}

function formatMessageForSummary(message) {
  if (message.role === 'user') {
    return `[user]\n${clipText(String(message.content || ''), 1200)}`
  }

  if (
    message.role === 'assistant' &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    const toolLines = message.tool_calls
      .map(
        (toolCall) =>
          `- ${toolCall.function.name}(${clipText(toolCall.function.arguments || '{}', 400)})`,
      )
      .join('\n')

    return `[assistant tool calls]\n${toolLines}`
  }

  if (message.role === 'assistant') {
    return `[assistant]\n${clipText(String(message.content || ''), 1200)}`
  }

  if (message.role === 'tool') {
    return `[tool result]\n${clipText(String(message.content || ''), 1200)}`
  }

  return ''
}

function buildFallbackMemory(existingMemory, archivedMessages) {
  const lines = archivedMessages.map(messageToMemoryLine).filter(Boolean)
  const merged = [existingMemory, ...lines].filter(Boolean).join('\n')
  return clipText(merged, MAX_MEMORY_TOKENS)
}

function estimateContextTokens(messages, memory) {
  return estimateTokensFromText(JSON.stringify(buildMessagesForModel(messages, memory))) +
    estimateTokensFromText(JSON.stringify(tools))
}

async function summarizeMemory(messages, existingMemory) {
  const fallback = buildFallbackMemory(existingMemory, messages)
  const historyText = messages.map(formatMessageForSummary).filter(Boolean).join('\n\n')

  try {
    const response = await client.chat.completions.create({
      model: SUMMARY_MODEL,
      messages: [
        {
          role: 'system',
          content:
            '你负责压缩对话上下文。请把历史消息总结成一段可继续推理的 memory，保留：用户目标、关键事实、已经调用过的工具与结果、已做出的决定、仍未完成的任务。不要写寒暄，不要重复无关细节。',
        },
        {
          role: 'user',
          content: `已有 memory：\n${existingMemory || '（无）'}\n\n请基于下面的历史消息，生成新的 memory。\n\n历史消息：\n${historyText}`,
        },
      ],
    })

    const summary = response.choices[0]?.message?.content?.trim()
    if (!summary) {
      return fallback
    }

    return clipText(summary, MAX_MEMORY_TOKENS)
  } catch (error) {
    console.error(`[context] summary failed: ${error.message}`)
    return fallback
  }
}

async function compactMessages(messages, memory) {
  if (estimateContextTokens(messages, memory.value) < SUMMARY_TRIGGER_TOKENS) {
    return
  }

  if (messages.length <= MAX_CONTEXT_MESSAGES + 1) {
    return
  }

  const archived = messages.slice(1, messages.length - MAX_CONTEXT_MESSAGES)
  if (archived.length === 0) {
    return
  }

  memory.value = await summarizeMemory(archived, memory.value)

  const recentMessages = messages.slice(-MAX_CONTEXT_MESSAGES)
  messages.splice(1, messages.length - 1, ...recentMessages)

  console.log(
    `[context] summarized ${archived.length} messages into memory, approx tokens=${estimateContextTokens(
      messages,
      memory.value,
    )}`,
  )
}

function buildMessagesForModel(messages, memory) {
  const baseMessages = [...messages]

  if (!memory) {
    return baseMessages
  }

  return [
    baseMessages[0],
    {
      role: baseMessages[0].role,
      content: `这是压缩后的历史上下文，只保留关键信息：\n${memory}`,
    },
    ...baseMessages.slice(1),
  ]
}

async function runAgent(userPrompt) {
  const memory = { value: '' }
  let messages = [
    {
      role: 'system',
      content:
        '你是一个教学演示用 agent。需要时调用工具，最后用中文给出简洁答案。',
    },
    {
      role: 'user',
      content: userPrompt,
    },
  ]

  for (let step = 1; step <= MAX_STEPS; step += 1) {
    console.log(`\n[loop ${step}] calling model...`)
    await compactMessages(messages, memory)

    const response = await client.chat.completions.create({
      model: MODEL,
      parallel_tool_calls: false,
      tools,
      messages: buildMessagesForModel(messages, memory.value),
    })

    const message = response.choices[0]?.message
    const toolCalls = message?.tool_calls ?? []

    if (toolCalls.length === 0) {
      console.log('\n[final answer]')
      console.log(message?.content || '')
      return
    }

    messages.push({
      role: 'assistant',
      content: message?.content || '',
      tool_calls: toolCalls,
    })

    for (const toolCall of toolCalls) {
      const toolName = toolCall.function.name
      const handler = toolHandlers[toolName]
      const args = JSON.parse(toolCall.function.arguments || '{}')

      console.log(`\n[tool] ${toolName}(${toolCall.function.arguments})`)

      let result
      try {
        result = await handler(args)
      } catch (error) {
        result = `Tool error: ${error.message}`
      }

      const output =
        typeof result === 'string' ? result : JSON.stringify(result, null, 2)
      const preview =
        output.length > 400 ? `${output.slice(0, 400)}\n...<truncated>` : output

      console.log(`[tool result]\n${preview}`)

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: clipText(output, MAX_TOOL_RESULT_TOKENS),
      })
    }
  }

  throw new Error(`超过最大循环次数: ${MAX_STEPS}`)
}

const prompt =
  process.argv.slice(2).join(' ') ||
  '请先列出当前目录下的文件，再读取 package.json，然后用中文总结这个项目里有哪些 scripts。'

runAgent(prompt).catch((error) => {
  console.error('\n[error]')
  console.error(error)
  process.exit(1)
})

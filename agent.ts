import 'dotenv/config'
import fs from 'node:fs/promises'
import path from 'node:path'
import OpenAI from 'openai'
import type {
  ChatCompletionFunctionTool,
  ChatCompletionTool,
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
  parseObjectArguments,
  readPromptFromCli,
  readNumericEnv,
  summarizeMemory,
  toErrorMessage,
  type AgentMessage,
  type MemoryState,
} from './agent-shared.js'

type ListFilesArgs = {
  dir?: string
}

type ReadFileArgs = {
  file_path: string
}

type ToolResult = string | FileEntry[]

type LocalToolDefinition<Args> = {
  definition: ChatCompletionFunctionTool
  parseArgs: (input: Record<string, unknown>) => Args
  execute: (args: Args) => Promise<ToolResult>
}

interface FileEntry {
  name: string
  type: 'dir' | 'file'
}

function requireSetting(value: string | undefined, message: string): string {
  if (value) {
    return value
  }

  console.error(message)
  process.exit(1)
}

const API_KEY = requireSetting(
  process.env.STEP_API_KEY ?? process.env.OPENAI_API_KEY,
  'Missing STEP_API_KEY or OPENAI_API_KEY',
)
const BASE_URL = process.env.STEP_BASE_URL ?? process.env.OPENAI_BASE_URL
const MODEL = requireSetting(
  process.env.STEP_MODEL ?? process.env.OPENAI_MODEL,
  'Missing STEP_MODEL or OPENAI_MODEL',
)
const SUMMARY_MODEL = process.env.SUMMARY_MODEL ?? MODEL
const ROOT_DIR = process.cwd()
const MAX_STEPS = 100
const MAX_CONTEXT_MESSAGES = readNumericEnv('MAX_CONTEXT_MESSAGES', 12)
const MAX_MEMORY_TOKENS = readNumericEnv(
  'MAX_MEMORY_TOKENS',
  Math.ceil(readNumericEnv('MAX_MEMORY_CHARS', 4000) / 3),
)
const MAX_TOOL_RESULT_TOKENS = readNumericEnv(
  'MAX_TOOL_RESULT_TOKENS',
  Math.ceil(readNumericEnv('MAX_TOOL_RESULT_CHARS', 4000) / 3),
)
const SUMMARY_TRIGGER_TOKENS = readNumericEnv(
  'SUMMARY_TRIGGER_TOKENS',
  Math.ceil(readNumericEnv('SUMMARY_TRIGGER_CHARS', 12000) / 3),
)

const client = new OpenAI({
  apiKey: API_KEY,
  baseURL: BASE_URL,
})

function resolveInsideRoot(target = '.'): string {
  const fullPath = path.resolve(ROOT_DIR, target)
  const relativePath = path.relative(ROOT_DIR, fullPath)

  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new Error('只能访问当前工作目录里的文件')
  }

  return fullPath
}

async function listFiles({ dir = '.' }: ListFilesArgs = {}): Promise<
  FileEntry[]
> {
  const fullPath = resolveInsideRoot(dir)
  const entries = await fs.readdir(fullPath, { withFileTypes: true })

  return entries.map((entry) => ({
    name: entry.name,
    type: entry.isDirectory() ? 'dir' : 'file',
  }))
}

async function readFile({ file_path }: ReadFileArgs): Promise<string> {
  const fullPath = resolveInsideRoot(file_path)
  const content = await fs.readFile(fullPath, 'utf8')

  if (content.length > 12000) {
    return `${content.slice(0, 12000)}\n...<truncated>`
  }

  return content
}

function parseListFilesArgs(input: Record<string, unknown>): ListFilesArgs {
  const dir = input.dir
  if (dir !== undefined && typeof dir !== 'string') {
    throw new Error('dir 必须是字符串')
  }

  return dir ? { dir } : {}
}

function parseReadFileArgs(input: Record<string, unknown>): ReadFileArgs {
  const filePath = input.file_path
  if (typeof filePath !== 'string' || filePath.length === 0) {
    throw new Error('file_path 不能为空')
  }

  return { file_path: filePath }
}

const listFilesTool: LocalToolDefinition<ListFilesArgs> = {
  definition: {
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
  parseArgs: parseListFilesArgs,
  execute: listFiles,
}

const readFileTool: LocalToolDefinition<ReadFileArgs> = {
  definition: {
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
  parseArgs: parseReadFileArgs,
  execute: readFile,
}

const tools: ChatCompletionTool[] = [
  listFilesTool.definition,
  readFileTool.definition,
]

async function executeTool(
  toolName: string,
  rawArguments: string,
): Promise<string> {
  const parsedArguments = parseObjectArguments(rawArguments)
  let result: ToolResult

  switch (toolName) {
    case 'list_files':
      result = await listFilesTool.execute(
        listFilesTool.parseArgs(parsedArguments),
      )
      break
    case 'read_file':
      result = await readFileTool.execute(
        readFileTool.parseArgs(parsedArguments),
      )
      break
    default:
      throw new Error(`未知工具: ${toolName}`)
  }

  return typeof result === 'string' ? result : JSON.stringify(result, null, 2)
}

async function runAgent(userPrompt: string): Promise<void> {
  const memory: MemoryState = { value: '' }
  const messages: AgentMessage[] = [
    createPromptMessage(
      'system',
      '你是一个教学演示用 agent。需要时调用工具，最后用中文给出简洁答案。',
    ),
    createUserMessage(userPrompt),
  ]

  for (let step = 1; step <= MAX_STEPS; step += 1) {
    console.log(`\n[loop ${step}] calling model...`)
    await compactMessages({
      messages,
      memory,
      tools,
      maxContextMessages: MAX_CONTEXT_MESSAGES,
      summaryTriggerTokens: SUMMARY_TRIGGER_TOKENS,
      summarize: (archivedMessages, existingMemory) =>
        summarizeMemory({
          client,
          summaryModel: SUMMARY_MODEL,
          messages: archivedMessages,
          existingMemory,
          maxMemoryTokens: MAX_MEMORY_TOKENS,
          instructionRole: 'system',
        }),
    })

    const response = await client.chat.completions.create({
      model: MODEL,
      parallel_tool_calls: false,
      tools,
      messages: buildMessagesForModel(messages, memory.value),
    })

    const message = response.choices[0]?.message
    if (!message) {
      throw new Error('模型没有返回消息')
    }

    const toolCalls = getFunctionToolCalls(message.tool_calls)
    const assistantContent =
      typeof message.content === 'string'
        ? message.content
        : message.refusal || ''

    if (toolCalls.length === 0) {
      console.log('\n[final answer]')
      console.log(assistantContent)
      return
    }

    messages.push(createAssistantToolCallMessage(assistantContent, toolCalls))

    for (const toolCall of toolCalls) {
      console.log(
        `\n[tool] ${toolCall.function.name}(${toolCall.function.arguments})`,
      )

      let output: string
      try {
        output = await executeTool(
          toolCall.function.name,
          toolCall.function.arguments,
        )
      } catch (error) {
        output = `Tool error: ${toErrorMessage(error)}`
      }

      const preview =
        output.length > 400 ? `${output.slice(0, 400)}\n...<truncated>` : output

      console.log(`[tool result]\n${preview}`)
      messages.push(
        createToolMessage(
          toolCall.id,
          clipText(output, MAX_TOOL_RESULT_TOKENS),
        ),
      )
    }
  }

  throw new Error(`超过最大循环次数: ${MAX_STEPS}`)
}

async function main(): Promise<void> {
  const prompt = await readPromptFromCli('请输入 prompt: ')
  await runAgent(prompt)
}

main().catch((error: unknown) => {
  console.error('\n[error]')
  console.error(toErrorMessage(error))
  process.exit(1)
})

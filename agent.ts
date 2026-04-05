import 'dotenv/config'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import OpenAI from 'openai'
import { Client as McpClient } from '@modelcontextprotocol/sdk/client/index.js'
import {
  StdioClientTransport,
  type StdioServerParameters,
} from '@modelcontextprotocol/sdk/client/stdio.js'
import {
  CallToolResultSchema,
  type CallToolResult,
  type CompatibilityCallToolResult,
  type Tool as McpTool,
} from '@modelcontextprotocol/sdk/types.js'
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
  inheritProcessEnv,
  normalizeMessageContent,
  parseObjectArguments,
  readPromptFromCli,
  readNumericEnv,
  summarizeMemory,
  toErrorMessage,
  type AgentMessage,
  type MemoryState,
  type PromptRole,
} from './agent-shared.js'

type ToolMode = 'local' | 'chrome'

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

interface ToolProvider {
  mode: ToolMode
  tools: ChatCompletionTool[]
  instructionRole: PromptRole
  systemPrompt: string
  maxSteps: number
  toolLogLabel: string
  resultLogLabel: string
  previewLimit: number
  executeTool: (toolName: string, rawArguments: string) => Promise<string>
  close: () => Promise<void>
}

interface CliOptions {
  mode: ToolMode
  promptArgs: string[]
}

type JsonSchemaObject = {
  $schema?: string
  type: 'object'
  properties?: Record<string, object>
  required?: string[]
  [key: string]: unknown
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
const CODEX_CONFIG_PATH = path.join(os.homedir(), '.codex', 'config.toml')
const LOCAL_MAX_STEPS = 100
const CHROME_MAX_STEPS = 10
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

function parseCliOptions(argv: string[]): CliOptions {
  let mode: ToolMode = 'local'
  const promptArgs: string[] = []

  for (const arg of argv) {
    if (arg === '--chrome') {
      mode = 'chrome'
      continue
    }

    promptArgs.push(arg)
  }

  return {
    mode,
    promptArgs,
  }
}

async function executeLocalTool(
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

function createLocalToolProvider(): ToolProvider {
  return {
    mode: 'local',
    tools,
    instructionRole: 'system',
    systemPrompt:
      '你是一个教学演示用 agent。需要时调用工具，最后用中文给出简洁答案。',
    maxSteps: LOCAL_MAX_STEPS,
    toolLogLabel: 'tool',
    resultLogLabel: 'tool result',
    previewLimit: 400,
    executeTool: executeLocalTool,
    close: async () => {},
  }
}

function readSection(text: string, header: string): string {
  const escapedHeader = header.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const sectionPattern = new RegExp(
    `\\[${escapedHeader}\\]\\n([\\s\\S]*?)(?=\\n\\[[^\\]]+\\]|$)`,
  )
  const match = text.match(sectionPattern)
  return match?.[1] ?? ''
}

function parseTomlString(section: string, key: string): string | null {
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = section.match(new RegExp(`^${escapedKey}\\s*=\\s*"(.*)"$`, 'm'))
  if (!match) {
    return null
  }

  const value = match[1]
  return value ? value.replace(/\\"/g, '"').replace(/\\\\/g, '\\') : null
}

function parseTomlStringArray(section: string, key: string): string[] {
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = section.match(new RegExp(`^${escapedKey}\\s*=\\s*\\[(.*?)\\]$`, 'ms'))
  if (!match) {
    return []
  }

  const rawArray = match[1]
  if (!rawArray) {
    return []
  }

  return Array.from(rawArray.matchAll(/"((?:\\.|[^"])*)"/g), (item) => {
    const value = item[1] ?? ''
    return value.replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  })
}

async function loadChromeMcpServerConfig(): Promise<StdioServerParameters> {
  const toml = await fs.readFile(CODEX_CONFIG_PATH, 'utf8')
  const serverSection = readSection(toml, 'mcp_servers.chrome-devtools')
  const envSection = readSection(toml, 'mcp_servers.chrome-devtools.env')

  const command = parseTomlString(serverSection, 'command')
  const args = parseTomlStringArray(serverSection, 'args')

  if (!command || args.length === 0) {
    throw new Error('在 ~/.codex/config.toml 里没有找到 chrome-devtools MCP 配置')
  }

  const config: StdioServerParameters = {
    command,
    args: [...args],
    env: inheritProcessEnv(),
  }

  const pathFromConfig = parseTomlString(envSection, 'PATH')
  if (pathFromConfig && config.env) {
    config.env.PATH = pathFromConfig
  }

  const demoFlags = ['--slim', '--headless', '--isolated']
  for (const flag of demoFlags) {
    if (!config.args?.includes(flag)) {
      config.args?.push(flag)
    }
  }

  return config
}

function sanitizeSchema(schema: unknown): JsonSchemaObject {
  if (!schema || typeof schema !== 'object') {
    return {
      type: 'object',
      properties: {},
    }
  }

  return JSON.parse(JSON.stringify(schema)) as JsonSchemaObject
}

function mcpToolsToOpenAITools(mcpTools: McpTool[]): ChatCompletionTool[] {
  return mcpTools.map(
    (tool): ChatCompletionFunctionTool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description || tool.title || `MCP tool: ${tool.name}`,
        parameters: sanitizeSchema(tool.inputSchema),
      },
    }),
  )
}

function formatMcpContentItem(item: CallToolResult['content'][number]): string {
  if (item.type === 'text') {
    return item.text
  }

  if (item.type === 'image') {
    return `[image ${item.mimeType}, base64 length=${item.data.length}]`
  }

  if (item.type === 'audio') {
    return `[audio ${item.mimeType}, base64 length=${item.data.length}]`
  }

  if (item.type === 'resource') {
    const { resource } = item
    if ('text' in resource) {
      return `[resource ${resource.uri}]\n${resource.text}`
    }

    return `[resource ${resource.uri}, blob length=${resource.blob.length}]`
  }

  return JSON.stringify(item, null, 2)
}

function hasContentBlocks(result: CompatibilityCallToolResult): result is CallToolResult {
  return Array.isArray((result as { content?: unknown }).content)
}

function formatMcpToolResult(result: CompatibilityCallToolResult): string {
  if (!hasContentBlocks(result)) {
    return JSON.stringify(
      {
        isError: false,
        structuredContent: null,
        content: [],
        toolResult: result.toolResult ?? null,
      },
      null,
      2,
    )
  }

  const payload = {
    isError: Boolean(result.isError),
    structuredContent: result.structuredContent || null,
    content: result.content.map(formatMcpContentItem),
  }

  return JSON.stringify(payload, null, 2)
}

async function createChromeToolProvider(): Promise<ToolProvider> {
  const serverConfig = await loadChromeMcpServerConfig()
  const transport = new StdioClientTransport({
    ...serverConfig,
    stderr: 'pipe',
  })

  if (transport.stderr) {
    transport.stderr.on('data', (chunk) => {
      const text = String(chunk).trim()
      if (text) {
        console.error(`[mcp stderr] ${text}`)
      }
    })
  }

  const mcpClient = new McpClient({
    name: 'simple-agent',
    version: '1.0.0',
  })

  mcpClient.onerror = (error) => {
    console.error('[mcp client error]', error)
  }

  try {
    await mcpClient.connect(transport)
    const toolResult = await mcpClient.listTools()
    const chromeTools = mcpToolsToOpenAITools(toolResult.tools)

    console.log('[mcp tools]')
    console.log(
      chromeTools
        .map((tool) => (tool.type === 'function' ? tool.function.name : tool.type))
        .join(', '),
    )

    return {
      mode: 'chrome',
      tools: chromeTools,
      instructionRole: 'developer',
      systemPrompt:
        '你是一个教学演示用 agent。你当前可以调用 Chrome DevTools MCP 工具。需要时调用工具，最后用中文给出简洁答案。',
      maxSteps: CHROME_MAX_STEPS,
      toolLogLabel: 'mcp tool',
      resultLogLabel: 'mcp result',
      previewLimit: 500,
      executeTool: async (toolName, rawArguments) => {
        const result = await mcpClient.callTool(
          {
            name: toolName,
            arguments: parseObjectArguments(rawArguments),
          },
          CallToolResultSchema,
        )

        return formatMcpToolResult(result)
      },
      close: async () => {
        await transport.close()
      },
    }
  } catch (error) {
    await transport.close().catch(() => {})
    throw error
  }
}

async function createToolProvider(mode: ToolMode): Promise<ToolProvider> {
  if (mode === 'chrome') {
    return createChromeToolProvider()
  }

  return createLocalToolProvider()
}

async function runAgent(userPrompt: string, provider: ToolProvider): Promise<void> {
  const memory: MemoryState = { value: '' }
  const messages: AgentMessage[] = [
    createPromptMessage(
      provider.instructionRole,
      provider.systemPrompt,
    ),
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

      const response = await client.chat.completions.create({
        model: MODEL,
        parallel_tool_calls: false,
        tools: provider.tools,
        messages: buildMessagesForModel(messages, memory.value),
      })

      const message = response.choices[0]?.message
      if (!message) {
        throw new Error('模型没有返回消息')
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

async function main(): Promise<void> {
  const cliOptions = parseCliOptions(process.argv.slice(2))
  const question =
    cliOptions.mode === 'chrome'
      ? '请输入 Chrome MCP prompt: '
      : '请输入 prompt: '
  const prompt = await readPromptFromCli(question, cliOptions.promptArgs)
  const provider = await createToolProvider(cliOptions.mode)
  await runAgent(prompt, provider)
}

main().catch((error: unknown) => {
  console.error('\n[error]')
  console.error(toErrorMessage(error))
  process.exit(1)
})

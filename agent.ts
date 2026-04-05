import 'dotenv/config'
import fs from 'node:fs/promises'
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

type ToolMode = 'local' | 'chrome' | 'combined'

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
  promptArgs: string[]
}

type JsonSchemaObject = {
  $schema?: string
  type: 'object'
  properties?: Record<string, object>
  required?: string[]
  [key: string]: unknown
}

type ChromeConnectionMode =
  | 'launch'
  | 'browser-url'
  | 'ws-endpoint'
  | 'auto-connect'

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
const CHROME_MCP_BIN_PATH = resolveInsideRoot(
  'node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js',
)
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

function readBooleanEnv(name: string, fallback: boolean): boolean {
  const rawValue = process.env[name]
  if (!rawValue) {
    return fallback
  }

  const normalized = rawValue.trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true
  }

  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false
  }

  return fallback
}

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
  return {
    promptArgs: argv.filter((arg) => arg !== '--chrome'),
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

function appendCliArg(
  args: string[],
  flag: string,
  value: string | undefined,
): void {
  if (!value) {
    return
  }

  args.push(flag, value)
}

function appendBooleanCliFlag(
  args: string[],
  flag: string,
  enabled: boolean,
): void {
  if (enabled) {
    args.push(flag)
  }
}

function assertChromeMcpNodeVersion(): void {
  const [majorRaw, minorRaw] = process.versions.node.split('.')
  const major = Number(majorRaw)
  const minor = Number(minorRaw)

  const isSupported =
    major > 22 || (major === 22 && minor >= 12) || (major === 20 && minor >= 19)

  if (!isSupported) {
    throw new Error(
      `Chrome MCP 工具要求 Node.js 20.19.0+ 或 22.12.0+。当前是 ${process.version}。请先升级 Node 再运行 npm start。`,
    )
  }
}

async function assertChromeMcpInstalled(): Promise<void> {
  try {
    await fs.access(CHROME_MCP_BIN_PATH)
  } catch {
    throw new Error(
      '未找到项目内置的 chrome-devtools-mcp。请先在项目根目录执行 npm install。',
    )
  }
}

function readChromeConnectionMode(): ChromeConnectionMode {
  if (process.env.CHROME_MCP_BROWSER_URL) {
    return 'browser-url'
  }

  if (process.env.CHROME_MCP_WS_ENDPOINT) {
    return 'ws-endpoint'
  }

  if (readBooleanEnv('CHROME_MCP_AUTO_CONNECT', false)) {
    return 'auto-connect'
  }

  return 'launch'
}

function buildChromeMcpArgs(): string[] {
  const args = [CHROME_MCP_BIN_PATH]
  const connectionMode = readChromeConnectionMode()

  appendBooleanCliFlag(args, '--slim', readBooleanEnv('CHROME_MCP_SLIM', true))

  if (connectionMode === 'browser-url') {
    appendCliArg(args, '--browserUrl', process.env.CHROME_MCP_BROWSER_URL)
  } else if (connectionMode === 'ws-endpoint') {
    appendCliArg(args, '--wsEndpoint', process.env.CHROME_MCP_WS_ENDPOINT)
    appendCliArg(args, '--wsHeaders', process.env.CHROME_MCP_WS_HEADERS)
  } else if (connectionMode === 'auto-connect') {
    appendBooleanCliFlag(args, '--autoConnect', true)
    appendCliArg(args, '--channel', process.env.CHROME_MCP_CHANNEL || 'stable')
  } else {
    appendBooleanCliFlag(
      args,
      '--headless',
      readBooleanEnv('CHROME_MCP_HEADLESS', true),
    )
    appendBooleanCliFlag(
      args,
      '--isolated',
      readBooleanEnv('CHROME_MCP_ISOLATED', true),
    )
    appendCliArg(args, '--channel', process.env.CHROME_MCP_CHANNEL)
    appendCliArg(
      args,
      '--executablePath',
      process.env.CHROME_MCP_EXECUTABLE_PATH,
    )
    appendCliArg(args, '--userDataDir', process.env.CHROME_MCP_USER_DATA_DIR)
  }

  appendCliArg(args, '--viewport', process.env.CHROME_MCP_VIEWPORT)
  appendCliArg(args, '--logFile', process.env.CHROME_MCP_LOG_FILE)
  appendCliArg(args, '--proxyServer', process.env.CHROME_MCP_PROXY_SERVER)

  if (readBooleanEnv('CHROME_MCP_ACCEPT_INSECURE_CERTS', false)) {
    args.push('--acceptInsecureCerts')
  }

  const chromeArgs = process.env.CHROME_MCP_CHROME_ARGS?.split(/\s*\n\s*/)
    .map((value) => value.trim())
    .filter(Boolean)
  if (chromeArgs) {
    for (const chromeArg of chromeArgs) {
      args.push('--chromeArg', chromeArg)
    }
  }

  const ignoreDefaultArgs = process.env.CHROME_MCP_IGNORE_DEFAULT_ARGS?.split(
    /\s*\n\s*/,
  )
    .map((value) => value.trim())
    .filter(Boolean)
  if (ignoreDefaultArgs) {
    for (const ignoredArg of ignoreDefaultArgs) {
      args.push('--ignoreDefaultChromeArg', ignoredArg)
    }
  }

  if (!readBooleanEnv('CHROME_MCP_USAGE_STATISTICS', true)) {
    args.push('--no-usage-statistics')
  }

  if (!readBooleanEnv('CHROME_MCP_PERFORMANCE_CRUX', true)) {
    args.push('--no-performance-crux')
  }

  return args
}

async function loadChromeMcpServerConfig(): Promise<StdioServerParameters> {
  assertChromeMcpNodeVersion()
  await assertChromeMcpInstalled()

  return {
    command: process.execPath,
    args: buildChromeMcpArgs(),
    env: inheritProcessEnv(),
    cwd: ROOT_DIR,
  }
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

function hasContentBlocks(
  result: CompatibilityCallToolResult,
): result is CallToolResult {
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
        .map((tool) =>
          tool.type === 'function' ? tool.function.name : tool.type,
        )
        .join(', '),
    )

    return {
      mode: 'chrome',
      tools: chromeTools,
      instructionRole: 'system',
      systemPrompt:
        '你是一个通用型agent。你当前可以调用 Chrome DevTools MCP 工具。需要时调用工具，最后用中文给出简洁答案。',
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

function collectFunctionToolNames(toolList: ChatCompletionTool[]): Set<string> {
  const names = new Set<string>()

  for (const tool of toolList) {
    if (tool.type !== 'function') {
      continue
    }
    names.add(tool.function.name)
  }

  return names
}

async function createCombinedToolProvider(): Promise<ToolProvider> {
  const localProvider = createLocalToolProvider()
  const chromeProvider = await createChromeToolProvider()

  const localToolNames = collectFunctionToolNames(localProvider.tools)
  const chromeToolNames = collectFunctionToolNames(chromeProvider.tools)

  for (const toolName of localToolNames) {
    if (chromeToolNames.has(toolName)) {
      throw new Error(`工具名冲突: ${toolName}`)
    }
  }

  return {
    mode: 'combined',
    tools: [...localProvider.tools, ...chromeProvider.tools],
    instructionRole: 'system',
    systemPrompt:
      '你是一个通用型agent。你可调用本地文件工具以及 Chrome DevTools MCP 工具。需要时调用工具，最后用中文给出简洁答案。',
    maxSteps: LOCAL_MAX_STEPS,
    toolLogLabel: 'tool',
    resultLogLabel: 'tool result',
    previewLimit: 500,
    executeTool: async (toolName, rawArguments) => {
      if (localToolNames.has(toolName)) {
        return localProvider.executeTool(toolName, rawArguments)
      }
      if (chromeToolNames.has(toolName)) {
        return chromeProvider.executeTool(toolName, rawArguments)
      }

      throw new Error(`未知工具: ${toolName}`)
    },
    close: async () => {
      await chromeProvider.close()
      await localProvider.close()
    },
  }
}

async function createToolProvider(): Promise<ToolProvider> {
  return createCombinedToolProvider()
}

async function runAgent(
  userPrompt: string,
  provider: ToolProvider,
): Promise<void> {
  const memory: MemoryState = { value: '' }
  const messages: AgentMessage[] = [
    createPromptMessage(provider.instructionRole, provider.systemPrompt),
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
  const prompt = await readPromptFromCli('请输入 prompt: ', cliOptions.promptArgs)
  const provider = await createToolProvider()
  await runAgent(prompt, provider)
}

main().catch((error: unknown) => {
  console.error('\n[error]')
  console.error(toErrorMessage(error))
  process.exit(1)
})

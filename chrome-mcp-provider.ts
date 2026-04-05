import fs from 'node:fs/promises'
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
  inheritProcessEnv,
  normalizeMessageContent,
  parseObjectArguments,
} from './shared.js'
import {
  CHROME_MCP_BIN_PATH,
  LOCAL_MAX_STEPS,
  ROOT_DIR,
  SUMMARY_MODEL,
  client,
  readBooleanEnv,
  readOptionalBooleanEnv,
} from './config.js'
import { executeLocalTool, localTools } from './local-tools.js'
import type {
  ChromeConnectionMode,
  JsonSchemaObject,
  ToolProvider,
} from './types.js'

type ChromeLaunchOverrides = {
  headless?: boolean
  isolated?: boolean
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

function appendRepeatedCliArgs(
  args: string[],
  flag: string,
  rawValues: string | undefined,
): void {
  const values = rawValues
    ?.split(/\s*\n\s*/)
    .map((value) => value.trim())
    .filter(Boolean)

  if (!values) {
    return
  }

  for (const value of values) {
    args.push(flag, value)
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

  if (readBooleanEnv('CHROME_MCP_AUTO_CONNECT', true)) {
    return 'auto-connect'
  }

  return 'launch'
}

async function inferChromeLaunchOverridesByModel(
  userPrompt: string,
): Promise<ChromeLaunchOverrides> {
  const response = await client.chat.completions.create({
    model: SUMMARY_MODEL,
    temperature: 0,
    messages: [
      {
        role: 'system',
        content:
          '你是浏览器启动参数判定器。只返回 JSON：{"headless": boolean, "isolated": boolean}。没有明确要求时都返回 false。',
      },
      {
        role: 'user',
        content: userPrompt,
      },
    ],
  })

  const content = normalizeMessageContent(response.choices[0]?.message?.content)
  if (!content) {
    return {}
  }

  try {
    const parsed = JSON.parse(content) as {
      headless?: unknown
      isolated?: unknown
    }

    const result: ChromeLaunchOverrides = {}
    if (typeof parsed.headless === 'boolean') {
      result.headless = parsed.headless
    }
    if (typeof parsed.isolated === 'boolean') {
      result.isolated = parsed.isolated
    }

    return result
  } catch {
    return {}
  }
}

async function buildChromeMcpArgs(userPrompt: string): Promise<string[]> {
  const args = [CHROME_MCP_BIN_PATH]
  const connectionMode = readChromeConnectionMode()
  const promptOverrides = await inferChromeLaunchOverridesByModel(userPrompt)

  const slimFlag = readOptionalBooleanEnv('CHROME_MCP_SLIM')
  appendBooleanCliFlag(args, '--slim', slimFlag === true)

  if (connectionMode === 'browser-url') {
    appendCliArg(args, '--browserUrl', process.env.CHROME_MCP_BROWSER_URL)
  } else if (connectionMode === 'ws-endpoint') {
    appendCliArg(args, '--wsEndpoint', process.env.CHROME_MCP_WS_ENDPOINT)
    appendCliArg(args, '--wsHeaders', process.env.CHROME_MCP_WS_HEADERS)
  } else if (connectionMode === 'auto-connect') {
    appendBooleanCliFlag(args, '--autoConnect', true)
    appendCliArg(args, '--channel', process.env.CHROME_MCP_CHANNEL || 'stable')
  } else {
    const headlessFlag =
      readOptionalBooleanEnv('CHROME_MCP_HEADLESS') ??
      promptOverrides.headless ??
      false
    const isolatedFlag =
      readOptionalBooleanEnv('CHROME_MCP_ISOLATED') ??
      promptOverrides.isolated ??
      false

    appendBooleanCliFlag(args, '--headless', headlessFlag)
    appendBooleanCliFlag(args, '--isolated', isolatedFlag)
    appendCliArg(args, '--channel', process.env.CHROME_MCP_CHANNEL)
    appendCliArg(args, '--executablePath', process.env.CHROME_MCP_EXECUTABLE_PATH)
    appendCliArg(args, '--userDataDir', process.env.CHROME_MCP_USER_DATA_DIR)
  }

  appendCliArg(args, '--viewport', process.env.CHROME_MCP_VIEWPORT)
  appendCliArg(args, '--logFile', process.env.CHROME_MCP_LOG_FILE)
  appendCliArg(args, '--proxyServer', process.env.CHROME_MCP_PROXY_SERVER)

  if (readBooleanEnv('CHROME_MCP_ACCEPT_INSECURE_CERTS', false)) {
    args.push('--acceptInsecureCerts')
  }

  appendRepeatedCliArgs(args, '--chromeArg', process.env.CHROME_MCP_CHROME_ARGS)
  appendRepeatedCliArgs(
    args,
    '--ignoreDefaultChromeArg',
    process.env.CHROME_MCP_IGNORE_DEFAULT_ARGS,
  )

  if (!readBooleanEnv('CHROME_MCP_USAGE_STATISTICS', true)) {
    args.push('--no-usage-statistics')
  }

  if (!readBooleanEnv('CHROME_MCP_PERFORMANCE_CRUX', true)) {
    args.push('--no-performance-crux')
  }

  return args
}

async function loadChromeMcpServerConfig(
  userPrompt: string,
): Promise<StdioServerParameters> {
  assertChromeMcpNodeVersion()
  await assertChromeMcpInstalled()
  const mcpArgs = await buildChromeMcpArgs(userPrompt)

  return {
    command: process.execPath,
    args: mcpArgs,
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

export async function createToolProvider(userPrompt: string): Promise<ToolProvider> {
  const serverConfig = await loadChromeMcpServerConfig(userPrompt)
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

    const localToolNames = collectFunctionToolNames(localTools)
    const chromeToolNames = collectFunctionToolNames(chromeTools)

    for (const toolName of localToolNames) {
      if (chromeToolNames.has(toolName)) {
        throw new Error(`工具名冲突: ${toolName}`)
      }
    }

    return {
      tools: [...localTools, ...chromeTools],
      instructionRole: 'system',
      maxSteps: LOCAL_MAX_STEPS,
      toolLogLabel: 'tool',
      resultLogLabel: 'tool result',
      previewLimit: 500,
      executeTool: async (toolName, rawArguments) => {
        if (localToolNames.has(toolName)) {
          return executeLocalTool(toolName, rawArguments)
        }
        if (chromeToolNames.has(toolName)) {
          const result = await mcpClient.callTool(
            {
              name: toolName,
              arguments: parseObjectArguments(rawArguments),
            },
            CallToolResultSchema,
          )
          return formatMcpToolResult(result)
        }

        throw new Error(
          `未知工具: ${toolName}。可用工具: ${[...localToolNames, ...chromeToolNames].join(', ')}`,
        )
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

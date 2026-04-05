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
  readNumericEnv,
  summarizeMemory,
  toErrorMessage,
  type AgentMessage,
  type MemoryState,
} from './agent-shared.js'

type JsonSchemaObject = {
  $schema?: string
  type: 'object'
  properties?: Record<string, object>
  required?: string[]
  [key: string]: unknown
}

const API_KEY = process.env.STEP_API_KEY ?? process.env.OPENAI_API_KEY
const BASE_URL =
  process.env.STEP_BASE_URL ??
  process.env.OPENAI_BASE_URL ??
  'https://api.stepfun.ai/step_plan/v1'
const MODEL = process.env.STEP_MODEL ?? process.env.OPENAI_MODEL ?? 'step-3.5-flash'
const SUMMARY_MODEL = process.env.SUMMARY_MODEL ?? MODEL
const MAX_STEPS = 10
const CODEX_CONFIG_PATH = path.join(os.homedir(), '.codex', 'config.toml')
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

function requireSetting(value: string | undefined, message: string): string {
  if (value) {
    return value
  }

  console.error(message)
  process.exit(1)
}

const openai = new OpenAI({
  apiKey: requireSetting(API_KEY, 'Missing STEP_API_KEY or OPENAI_API_KEY'),
  baseURL: BASE_URL,
})

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

async function runAgent(userPrompt: string): Promise<void> {
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

  await mcpClient.connect(transport)

  try {
    const toolResult = await mcpClient.listTools()
    const tools = mcpToolsToOpenAITools(toolResult.tools)

    console.log('[mcp tools]')
    console.log(
      tools
        .map((tool) => (tool.type === 'function' ? tool.function.name : tool.type))
        .join(', '),
    )

    const memory: MemoryState = { value: '' }
    const messages: AgentMessage[] = [
      createPromptMessage(
        'developer',
        '你是一个教学演示用 agent。你当前可以调用 Chrome DevTools MCP 工具。需要时调用工具，最后用中文给出简洁答案。',
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
            client: openai,
            summaryModel: SUMMARY_MODEL,
            messages: archivedMessages,
            existingMemory,
            maxMemoryTokens: MAX_MEMORY_TOKENS,
            instructionRole: 'developer',
          }),
      })

      const response = await openai.chat.completions.create({
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
        normalizeMessageContent(message.content) || message.refusal || ''

      if (toolCalls.length === 0) {
        console.log('\n[final answer]')
        console.log(assistantContent)
        return
      }

      messages.push(createAssistantToolCallMessage(assistantContent, toolCalls))

      for (const toolCall of toolCalls) {
        console.log(`\n[mcp tool] ${toolCall.function.name}(${toolCall.function.arguments})`)

        let output: string
        try {
          const result = await mcpClient.callTool({
            name: toolCall.function.name,
            arguments: parseObjectArguments(toolCall.function.arguments),
          }, CallToolResultSchema)
          output = formatMcpToolResult(result)
        } catch (error) {
          output = `MCP tool error: ${toErrorMessage(error)}`
        }

        const preview =
          output.length > 500 ? `${output.slice(0, 500)}\n...<truncated>` : output
        console.log(`[mcp result]\n${preview}`)

        messages.push(
          createToolMessage(toolCall.id, clipText(output, MAX_TOOL_RESULT_TOKENS)),
        )
      }
    }

    throw new Error(`超过最大循环次数: ${MAX_STEPS}`)
  } finally {
    await transport.close()
  }
}

const prompt =
  process.argv.slice(2).join(' ') ||
  '打开 https://example.com ，读取页面标题，然后用中文告诉我标题是什么。'

runAgent(prompt).catch((error: unknown) => {
  console.error('\n[error]')
  console.error(toErrorMessage(error))
  process.exit(1)
})

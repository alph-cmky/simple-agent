import path from 'node:path'
import OpenAI from 'openai'
import { readNumericEnv } from './shared.js'

const BOOL_TRUE_SET = new Set(['1', 'true', 'yes', 'on'])
const BOOL_FALSE_SET = new Set(['0', 'false', 'no', 'off'])

export const SYSTEM_PROMPT =
  '你是一个通用型agent。你可调用本地文件工具以及 Chrome DevTools MCP 工具。默认优先复用当前浏览器会话与登录态，不要主动创建隔离上下文；仅在用户明确要求时才使用隔离方式。若用户指定了账号（如“使用某个 Google 账号”），先在目标站点检查当前登录账号是否匹配，再继续执行任务；若不匹配，先明确告知并给出下一步操作建议。严格只调用工具列表中存在的工具名；元素 uid 仅使用形如 "1_40" 的值，不要传 "uid=1_40"。在网页任务中优先少量高价值调用，避免反复滚动或重复快照。需要时调用工具，最后用中文给出简洁答案。今天是2026年4月5日。'

function requireSetting(value: string | undefined, message: string): string {
  if (value) {
    return value
  }

  console.error(message)
  process.exit(1)
}

function tokensFromChars(name: string, fallbackChars: number): number {
  return Math.ceil(readNumericEnv(name, fallbackChars) / 3)
}

export const ROOT_DIR = process.cwd()

export const API_KEY = requireSetting(
  process.env.API_KEY ?? process.env.OPENAI_API_KEY,
  'Missing API_KEY or OPENAI_API_KEY',
)
export const BASE_URL = process.env.BASE_URL ?? process.env.OPENAI_BASE_URL
export const MODEL = requireSetting(
  process.env.MODEL ?? process.env.OPENAI_MODEL,
  'Missing MODEL or OPENAI_MODEL',
)
export const SUMMARY_MODEL = process.env.SUMMARY_MODEL ?? MODEL

export const LOCAL_MAX_STEPS = 100
export const MAX_CONTEXT_MESSAGES = readNumericEnv('MAX_CONTEXT_MESSAGES', 100)
export const MAX_MEMORY_TOKENS = readNumericEnv(
  'MAX_MEMORY_TOKENS',
  tokensFromChars('MAX_MEMORY_CHARS', 3000),
)
export const MAX_TOOL_RESULT_TOKENS = readNumericEnv(
  'MAX_TOOL_RESULT_TOKENS',
  tokensFromChars('MAX_TOOL_RESULT_CHARS', 1500),
)
export const SUMMARY_TRIGGER_TOKENS = readNumericEnv(
  'SUMMARY_TRIGGER_TOKENS',
  tokensFromChars('SUMMARY_TRIGGER_CHARS', 30000),
)

export const RATE_LIMIT_RETRY_DELAYS_MS = [1200, 2500]

export function readBooleanEnv(name: string, fallback: boolean): boolean {
  const rawValue = process.env[name]
  if (!rawValue) {
    return fallback
  }

  const normalized = rawValue.trim().toLowerCase()
  if (BOOL_TRUE_SET.has(normalized)) {
    return true
  }

  if (BOOL_FALSE_SET.has(normalized)) {
    return false
  }

  return fallback
}

export function readOptionalBooleanEnv(name: string): boolean | undefined {
  const rawValue = process.env[name]
  if (!rawValue) {
    return undefined
  }

  const normalized = rawValue.trim().toLowerCase()
  if (BOOL_TRUE_SET.has(normalized)) {
    return true
  }

  if (BOOL_FALSE_SET.has(normalized)) {
    return false
  }

  return undefined
}

export function resolveInsideRoot(target = '.'): string {
  const fullPath = path.resolve(ROOT_DIR, target)
  const relativePath = path.relative(ROOT_DIR, fullPath)

  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new Error('只能访问当前工作目录里的文件')
  }

  return fullPath
}

export const CHROME_MCP_BIN_PATH = resolveInsideRoot(
  'node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js',
)

export const client = new OpenAI({
  apiKey: API_KEY,
  baseURL: BASE_URL,
})

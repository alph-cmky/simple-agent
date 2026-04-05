import type { ChatCompletionTool } from 'openai/resources/chat/completions'
import type { PromptRole } from './shared.js'

export interface ToolProvider {
  tools: ChatCompletionTool[]
  instructionRole: PromptRole
  maxSteps: number
  toolLogLabel: string
  resultLogLabel: string
  previewLimit: number
  executeTool: (toolName: string, rawArguments: string) => Promise<string>
  close: () => Promise<void>
}

export type JsonSchemaObject = {
  $schema?: string
  type: 'object'
  properties?: Record<string, object>
  required?: string[]
  [key: string]: unknown
}

export type ChromeConnectionMode =
  | 'launch'
  | 'browser-url'
  | 'ws-endpoint'
  | 'auto-connect'

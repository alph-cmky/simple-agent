import fs from 'node:fs/promises'
import type {
  ChatCompletionFunctionTool,
  ChatCompletionTool,
} from 'openai/resources/chat/completions'
import { parseObjectArguments } from './shared.js'
import { resolveInsideRoot } from './config.js'

type ListFilesArgs = {
  dir?: string
}

type ReadFileArgs = {
  file_path: string
}

type FileEntry = {
  name: string
  type: 'dir' | 'file'
}

type ToolResult = string | FileEntry[]

type LocalToolDefinition<Args> = {
  definition: ChatCompletionFunctionTool
  parseArgs: (input: Record<string, unknown>) => Args
  execute: (args: Args) => Promise<ToolResult>
}

async function listFiles({ dir = '.' }: ListFilesArgs = {}): Promise<FileEntry[]> {
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

export const localTools: ChatCompletionTool[] = [
  listFilesTool.definition,
  readFileTool.definition,
]

export async function executeLocalTool(
  toolName: string,
  rawArguments: string,
): Promise<string> {
  const parsedArguments = parseObjectArguments(rawArguments)

  const executors: Record<string, () => Promise<ToolResult>> = {
    list_files: () =>
      listFilesTool.execute(listFilesTool.parseArgs(parsedArguments)),
    read_file: () => readFileTool.execute(readFileTool.parseArgs(parsedArguments)),
  }

  const executor = executors[toolName]
  if (!executor) {
    throw new Error(`未知工具: ${toolName}。可用本地工具: list_files, read_file`)
  }

  const result = await executor()
  return typeof result === 'string' ? result : JSON.stringify(result, null, 2)
}

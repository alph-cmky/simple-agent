import 'dotenv/config'
import { readPromptFromCli, toErrorMessage } from './shared.js'
import { createToolProvider } from './chrome-mcp-provider.js'
import { runAgent } from './runner.js'

type CliOptions = {
  promptArgs: string[]
}

function parseCliOptions(argv: string[]): CliOptions {
  return {
    promptArgs: argv.filter((arg) => arg !== '--chrome'),
  }
}

async function main(): Promise<void> {
  const cliOptions = parseCliOptions(process.argv.slice(2))
  const prompt = await readPromptFromCli('请输入 prompt: ', cliOptions.promptArgs)
  const provider = await createToolProvider(prompt)
  await runAgent(prompt, provider)
}

main().catch((error: unknown) => {
  console.error('\n[error]')
  console.error(toErrorMessage(error))
  process.exit(1)
})

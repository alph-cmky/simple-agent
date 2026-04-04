import "dotenv/config";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import OpenAI from "openai";
import { Client as McpClient } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const API_KEY = process.env.STEP_API_KEY || process.env.OPENAI_API_KEY;
const BASE_URL =
  process.env.STEP_BASE_URL ||
  process.env.OPENAI_BASE_URL ||
  "https://api.stepfun.ai/step_plan/v1";
const MODEL =
  process.env.STEP_MODEL || process.env.OPENAI_MODEL || "step-3.5-flash";
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || MODEL;
const MAX_STEPS = 10;
const CODEX_CONFIG_PATH = path.join(os.homedir(), ".codex", "config.toml");
const MAX_CONTEXT_MESSAGES = Number(process.env.MAX_CONTEXT_MESSAGES || 12);
const MAX_MEMORY_TOKENS = Number(
  process.env.MAX_MEMORY_TOKENS ||
    Math.ceil(Number(process.env.MAX_MEMORY_CHARS || 4000) / 3)
);
const MAX_TOOL_RESULT_TOKENS = Number(
  process.env.MAX_TOOL_RESULT_TOKENS ||
    Math.ceil(Number(process.env.MAX_TOOL_RESULT_CHARS || 4000) / 3)
);
const SUMMARY_TRIGGER_TOKENS = Number(
  process.env.SUMMARY_TRIGGER_TOKENS ||
    Math.ceil(Number(process.env.SUMMARY_TRIGGER_CHARS || 12000) / 3)
);

if (!API_KEY) {
  console.error("Missing STEP_API_KEY or OPENAI_API_KEY");
  process.exit(1);
}

const openai = new OpenAI({
  apiKey: API_KEY,
  baseURL: BASE_URL
});

function readSection(text, header) {
  const escapedHeader = header.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const sectionPattern = new RegExp(`\\[${escapedHeader}\\]\\n([\\s\\S]*?)(?=\\n\\[[^\\]]+\\]|$)`);
  const match = text.match(sectionPattern);
  return match ? match[1] : "";
}

function parseTomlString(section, key) {
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = section.match(new RegExp(`^${escapedKey}\\s*=\\s*"(.*)"$`, "m"));
  if (!match) {
    return null;
  }

  return match[1]
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\");
}

function parseTomlStringArray(section, key) {
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = section.match(new RegExp(`^${escapedKey}\\s*=\\s*\\[(.*?)\\]$`, "ms"));

  if (!match) {
    return [];
  }

  return Array.from(match[1].matchAll(/"((?:\\.|[^"])*)"/g), (item) =>
    item[1].replace(/\\"/g, '"').replace(/\\\\/g, "\\")
  );
}

async function loadChromeMcpServerConfig() {
  const toml = await fs.readFile(CODEX_CONFIG_PATH, "utf8");
  const serverSection = readSection(toml, "mcp_servers.chrome-devtools");
  const envSection = readSection(toml, "mcp_servers.chrome-devtools.env");

  const command = parseTomlString(serverSection, "command");
  const args = parseTomlStringArray(serverSection, "args");

  if (!command || args.length === 0) {
    throw new Error("在 ~/.codex/config.toml 里没有找到 chrome-devtools MCP 配置");
  }

  const config = {
    command,
    args: [...args],
    env: {
      ...process.env
    }
  };

  const pathFromConfig = parseTomlString(envSection, "PATH");
  if (pathFromConfig) {
    config.env.PATH = pathFromConfig;
  }

  const demoFlags = ["--slim", "--headless", "--isolated"];
  for (const flag of demoFlags) {
    if (!config.args.includes(flag)) {
      config.args.push(flag);
    }
  }

  return config;
}

function sanitizeSchema(schema) {
  if (!schema || typeof schema !== "object") {
    return {
      type: "object",
      properties: {}
    };
  }

  return JSON.parse(JSON.stringify(schema));
}

function mcpToolsToOpenAITools(mcpTools) {
  return mcpTools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description || tool.title || `MCP tool: ${tool.name}`,
      parameters: sanitizeSchema(tool.inputSchema)
    }
  }));
}

function formatMcpContentItem(item) {
  if (item.type === "text") {
    return item.text;
  }

  if (item.type === "image") {
    return `[image ${item.mimeType}, base64 length=${item.data.length}]`;
  }

  if (item.type === "audio") {
    return `[audio ${item.mimeType}, base64 length=${item.data.length}]`;
  }

  if (item.type === "resource") {
    const resource = item.resource;
    if ("text" in resource) {
      return `[resource ${resource.uri}]\n${resource.text}`;
    }
    return `[resource ${resource.uri}, blob length=${resource.blob.length}]`;
  }

  return JSON.stringify(item, null, 2);
}

function formatMcpToolResult(result) {
  const payload = {
    isError: Boolean(result.isError),
    structuredContent: result.structuredContent || null,
    content: (result.content || []).map(formatMcpContentItem)
  };

  return JSON.stringify(payload, null, 2);
}

function estimateTokensFromText(text) {
  if (!text) {
    return 0;
  }

  const cjkMatches =
    text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [];
  const nonCjk = text.replace(
    /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu,
    ""
  );
  const nonWhitespaceChars = nonCjk.replace(/\s+/g, "").length;

  return cjkMatches.length + Math.ceil(nonWhitespaceChars / 4);
}

function clipText(text, maxTokens) {
  if (estimateTokensFromText(text) <= maxTokens) {
    return text;
  }

  const suffix = "\n...<truncated>";
  const allowedTokens = Math.max(1, maxTokens - estimateTokensFromText(suffix));
  let left = 0;
  let right = text.length;
  let best = "";

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const candidate = text.slice(0, mid);

    if (estimateTokensFromText(candidate) <= allowedTokens) {
      best = candidate;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return `${best}${suffix}`;
}

function messageToMemoryLine(message) {
  if (message.role === "user") {
    return `User: ${clipText(String(message.content || ""), 240)}`;
  }

  if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    const names = message.tool_calls.map((toolCall) => toolCall.function.name).join(", ");
    return `Assistant called tools: ${names}`;
  }

  if (message.role === "assistant") {
    return `Assistant: ${clipText(String(message.content || ""), 240)}`;
  }

  if (message.role === "tool") {
    return `Tool result: ${clipText(String(message.content || ""), 240)}`;
  }

  return "";
}

function formatMessageForSummary(message) {
  if (message.role === "user") {
    return `[user]\n${clipText(String(message.content || ""), 1200)}`;
  }

  if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    const toolLines = message.tool_calls
      .map(
        (toolCall) =>
          `- ${toolCall.function.name}(${clipText(toolCall.function.arguments || "{}", 400)})`
      )
      .join("\n");

    return `[assistant tool calls]\n${toolLines}`;
  }

  if (message.role === "assistant") {
    return `[assistant]\n${clipText(String(message.content || ""), 1200)}`;
  }

  if (message.role === "tool") {
    return `[tool result]\n${clipText(String(message.content || ""), 1200)}`;
  }

  return "";
}

function buildFallbackMemory(existingMemory, archivedMessages) {
  const lines = archivedMessages.map(messageToMemoryLine).filter(Boolean);
  const merged = [existingMemory, ...lines].filter(Boolean).join("\n");
  return clipText(merged, MAX_MEMORY_TOKENS);
}

function estimateContextTokens(messages, memory, tools) {
  return (
    estimateTokensFromText(JSON.stringify(buildMessagesForModel(messages, memory))) +
    estimateTokensFromText(JSON.stringify(tools))
  );
}

async function summarizeMemory(messages, existingMemory) {
  const fallback = buildFallbackMemory(existingMemory, messages);
  const historyText = messages.map(formatMessageForSummary).filter(Boolean).join("\n\n");

  try {
    const response = await openai.chat.completions.create({
      model: SUMMARY_MODEL,
      messages: [
        {
          role: "developer",
          content:
            "你负责压缩对话上下文。请把历史消息总结成一段可继续推理的 memory，保留：用户目标、关键事实、已经调用过的工具与结果、已做出的决定、仍未完成的任务。不要写寒暄，不要重复无关细节。"
        },
        {
          role: "user",
          content: `已有 memory：\n${existingMemory || "（无）"}\n\n请基于下面的历史消息，生成新的 memory。\n\n历史消息：\n${historyText}`
        }
      ]
    });

    const summary = response.choices[0]?.message?.content?.trim();
    if (!summary) {
      return fallback;
    }

    return clipText(summary, MAX_MEMORY_TOKENS);
  } catch (error) {
    console.error(`[context] summary failed: ${error.message}`);
    return fallback;
  }
}

async function compactMessages(messages, memory, tools) {
  if (estimateContextTokens(messages, memory.value, tools) < SUMMARY_TRIGGER_TOKENS) {
    return;
  }

  if (messages.length <= MAX_CONTEXT_MESSAGES + 1) {
    return;
  }

  const archived = messages.slice(1, messages.length - MAX_CONTEXT_MESSAGES);
  if (archived.length === 0) {
    return;
  }

  memory.value = await summarizeMemory(archived, memory.value);

  const recentMessages = messages.slice(-MAX_CONTEXT_MESSAGES);
  messages.splice(1, messages.length - 1, ...recentMessages);

  console.log(
    `[context] summarized ${archived.length} messages into memory, approx tokens=${estimateContextTokens(
      messages,
      memory.value,
      tools
    )}`
  );
}

function buildMessagesForModel(messages, memory) {
  const baseMessages = [...messages];

  if (!memory) {
    return baseMessages;
  }

  return [
    baseMessages[0],
    {
      role: baseMessages[0].role,
      content: `这是压缩后的历史上下文，只保留关键信息：\n${memory}`
    },
    ...baseMessages.slice(1)
  ];
}

async function runAgent(userPrompt) {
  const serverConfig = await loadChromeMcpServerConfig();
  const transport = new StdioClientTransport({
    ...serverConfig,
    stderr: "pipe"
  });

  if (transport.stderr) {
    transport.stderr.on("data", (chunk) => {
      const text = String(chunk).trim();
      if (text) {
        console.error(`[mcp stderr] ${text}`);
      }
    });
  }

  const mcpClient = new McpClient({
    name: "simple-agent",
    version: "1.0.0"
  });

  mcpClient.onerror = (error) => {
    console.error("[mcp client error]", error);
  };

  await mcpClient.connect(transport);

  try {
    const toolResult = await mcpClient.listTools();
    const tools = mcpToolsToOpenAITools(toolResult.tools);

    console.log("[mcp tools]");
    console.log(tools.map((tool) => tool.function.name).join(", "));

    const memory = { value: "" };
    let messages = [
      {
        role: "developer",
        content:
          "你是一个教学演示用 agent。你当前可以调用 Chrome DevTools MCP 工具。需要时调用工具，最后用中文给出简洁答案。"
      },
      {
        role: "user",
        content: userPrompt
      }
    ];

    for (let step = 1; step <= MAX_STEPS; step += 1) {
      console.log(`\n[loop ${step}] calling model...`);
      await compactMessages(messages, memory, tools);

      const response = await openai.chat.completions.create({
        model: MODEL,
        parallel_tool_calls: false,
        tools,
        messages: buildMessagesForModel(messages, memory.value)
      });

      const message = response.choices[0]?.message;
      const toolCalls = message?.tool_calls ?? [];

      if (toolCalls.length === 0) {
        console.log("\n[final answer]");
        console.log(message?.content || "");
        return;
      }

      messages.push({
        role: "assistant",
        content: message?.content || "",
        tool_calls: toolCalls
      });

      for (const toolCall of toolCalls) {
        const args = JSON.parse(toolCall.function.arguments || "{}");

        console.log(`\n[mcp tool] ${toolCall.function.name}(${toolCall.function.arguments})`);

        let output;
        try {
          const result = await mcpClient.callTool({
            name: toolCall.function.name,
            arguments: args
          });
          output = formatMcpToolResult(result);
        } catch (error) {
          output = `MCP tool error: ${error.message}`;
        }

        const preview = output.length > 500 ? `${output.slice(0, 500)}\n...<truncated>` : output;
        console.log(`[mcp result]\n${preview}`);

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: clipText(output, MAX_TOOL_RESULT_TOKENS)
        });
      }
    }

    throw new Error(`超过最大循环次数: ${MAX_STEPS}`);
  } finally {
    await transport.close();
  }
}

const prompt =
  process.argv.slice(2).join(" ") ||
  "打开 https://example.com ，读取页面标题，然后用中文告诉我标题是什么。";

runAgent(prompt).catch((error) => {
  console.error("\n[error]");
  console.error(error);
  process.exit(1);
});

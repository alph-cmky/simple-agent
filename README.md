# Simple Agent

一个尽量简单的 TypeScript agent loop 示例。

它只做 4 件事：

1. 把用户问题发给模型。
2. 如果模型请求调用本地 tool，就在本机执行。
3. 把 tool 结果再传回模型。
4. 直到模型不再要 tool，输出最终答案。

## 安装

```bash
cd simple-agent
npm install
```

## 配置

```bash
cp .env.example .env
```

然后把 `.env` 里的 key 改成你自己的 Step API Key。

当前示例优先读取这几个环境变量：

- `STEP_API_KEY`
- `STEP_BASE_URL`
- `STEP_MODEL`

同时也兼容：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## 运行

```bash
npm start -- "请列出当前目录文件，再读取 package.json，最后告诉我 scripts 是什么"
```

如果你不传 prompt，脚本会用内置的一条默认问题。

## 代码里最关键的部分

- `tools`：告诉模型“你有哪些本地能力可用”。
- `toolDefinitions / executeTool`：把 tool 名字映射到本地 TS 函数。
- `message.tool_calls`：拿到模型返回的内容，判断有没有要调用的 tool。
- `role: "tool"`：把本地 tool 执行结果作为 tool message 喂回模型，进入下一轮。

## 简单上下文管理

这版补了一个更实用的上下文管理层，规则是：

1. 永远保留第一条提示词。
2. 最近几条消息原样保留。
3. 当上下文体积快到阈值时，把更早的消息发给模型生成新的 memory。
4. 新 memory 会和旧 memory 合并，再作为一条额外提示词发给模型。
5. tool 输出先截断，再放回上下文。

默认参数：

- `MAX_CONTEXT_MESSAGES=12`：最近保留 12 条原始消息
- `MAX_MEMORY_TOKENS`：压缩 memory 的近似 token 上限
- `MAX_TOOL_RESULT_TOKENS`：单次 tool 结果回填时的近似 token 上限
- `SUMMARY_TRIGGER_TOKENS`：估算上下文接近这个近似 token 数时触发摘要压缩
- `SUMMARY_MODEL`：可选，单独指定生成 memory 的模型；默认和主模型相同

近似 token 算法很简单：

- 中日韩字符按 `1 token` 估算
- 其他非空白字符按 `4 个字符约 1 token` 估算

这样不需要额外 tokenizer 依赖，但比单纯按字符数更接近真实上下文大小。

可以这样改：

```bash
MAX_CONTEXT_MESSAGES=20 MAX_MEMORY_TOKENS=1500 SUMMARY_TRIGGER_TOKENS=6000 npm start
```

## 当前示例里的本地 tool

- `list_files(dir)`：列出目录内容
- `read_file(file_path)`：读取文本文件

这两个 tool 足够看懂最基本的 agent loop 是怎么跑起来的。

## 接本地 MCP 工具

如果你本机装了 MCP server，也可以把它接进同样的 agent loop。

这个目录里额外提供了一个 Chrome DevTools MCP 示例：

- [chrome-mcp-agent.ts](/Users/gaoyinrun/Desktop/qy/simple-agent/chrome-mcp-agent.ts)

它做的事情是：

1. 从 `~/.codex/config.toml` 读取 `chrome-devtools` 的本地配置。
2. 用 `@modelcontextprotocol/sdk` 通过 stdio 连接 MCP server。
3. 调用 `listTools()` 取到 MCP tools。
4. 把这些 MCP tools 转成 Chat Completions 可调用的 function tools。
5. 模型发起 tool call 后，再转发给 MCP server 执行。

### 安装额外依赖

```bash
npm install
```

### 运行 Chrome MCP 版本

```bash
npm run start:chrome
```

或者带一条 prompt：

```bash
npm run start:chrome -- "打开 https://example.com ，读取页面标题，然后告诉我页面标题"
```

### 这个版本和前一个版本的区别

- `agent.ts`：tool 是你自己在 TS 里手写的本地函数
- `chrome-mcp-agent.ts`：tool 来自本机 MCP server

本质上 loop 没变，变的是“tool 的执行端”。

### 注意

- 这个示例默认会给 `chrome-devtools-mcp` 追加 `--slim --headless --isolated`
- 这样更适合学习：工具少、不会干扰你平时打开的 Chrome
- 如果你的 `~/.codex/config.toml` 里没有 `chrome-devtools` 配置，这个脚本会直接报错
- 你的本机 Node / npm 版本最好至少是 `18.14.1`，否则部分 MCP 依赖可能只给 warning 或出现兼容性问题

## TypeScript 开发

```bash
npm run check
npm run build
```

- `npm run check`：执行 TypeScript 类型检查
- `npm run build`：把 `.ts` 编译到 `dist/`

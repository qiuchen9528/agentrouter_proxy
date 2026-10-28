# AgentRouter Proxy for OpenCode

本地双协议代理：让 [OpenCode](https://github.com/sst/opencode) 通过 [AgentRouter](https://agentrouter.org) 正常使用流式输出、工具调用与 Todo 面板。

同时兼容：

| 客户端 SDK | baseURL | 入口路径 |
|---|---|---|
| `@ai-sdk/anthropic` | `http://127.0.0.1:7187` | `POST /v1/messages` |
| `@ai-sdk/openai-compatible` | `http://127.0.0.1:7187/v1` | `POST /v1/chat/completions` |

---

## 为什么需要这个代理

AgentRouter 对客户端有 WAF / 指纹校验：

- Node.js `fetch`、原始 `httpx` 等常见客户端容易被拒（`unauthorized client detected`）
- 官方 Python **同步** Anthropic SDK 可以通过校验

OpenCode 侧基于 AI SDK，对 SSE 事件顺序、工具参数 JSON、`finish_reason` / `stop_reason` 非常严格。上游偶发的乱序、截断 JSON、`ping` 插入或错误的 `end_turn`，都会导致：

- 过程输出卡住或空白
- `todowrite` 解析失败、Todo 列表被清空
- 工具卡片不出现或无法执行

本代理在本地：

1. 用可过 WAF 的方式转发请求
2. 把上游 SSE 修成 AI SDK 能稳定解析的形态
3. 在一条进程里同时提供 Anthropic 与 OpenAI Compatible 两种接口

---

## 功能特性

### Anthropic（`/v1/messages`）

- 通过 Python Anthropic SDK 访问上游
- 保证 `message_start → content_block_* → message_delta → message_stop` 顺序
- 工具块尽早发出 `content_block_start`（工具卡片可立即出现）
- 缓冲并尽量修复截断的 tool input JSON（不轻易替换成 `{}`）
- 有工具时把错误的 `end_turn` 覆盖为 `tool_use`
- 转发 `ping`，不中断工具流
- 剥离会触发上游 400 的字段（如 adaptive thinking、`output_config`、块级 `cache_control`）
- 透传 `anthropic-beta` / `anthropic-version`

### OpenAI Compatible（`/v1/chat/completions`）

- 输出标准 `chat.completion.chunk` + 结尾 `data: [DONE]`
- 工具：先发 `name` / `id`，再刷完整 `arguments`；有工具时 `finish_reason` 为 `tool_calls`
- 支持 `reasoning_content` / `reasoning`，以及内容里的 `<think>...</think>` 拆分到过程输出
- 非流式响应同样规范化 tool_calls 与 finish_reason

### OpenAI 上游策略（`AGENTROUTER_OPENAI_UPSTREAM`）

| 值 | 行为 |
|---|---|
| `auto`（默认） | 先直连 OpenAI 网关；若 401/403 unauthorized，自动改走 Anthropic 再转回 OpenAI chunk |
| `openai` | 只走 OpenAI 网关 |
| `anthropic` | 始终把 OpenAI 请求翻译成 Anthropic，再映射回 OpenAI 格式 |

---

## 依赖

```text
python >= 3.10
fastapi
uvicorn
anthropic
httpx2          # 与项目环境一致的 HTTP 客户端（TLS / 连接池）
```

安装示例：

```bash
pip install fastapi uvicorn anthropic
# httpx2 按你本机/项目环境安装；若仅有 httpx，需自行把 import 改为 httpx
```

---

## 快速开始

```bash
export AGENTROUTER_API_KEY='sk-你的AgentRouter密钥'

# 可选
export AGENTROUTER_BASE_URL='https://agentrouter.org'
export AGENTROUTER_HOST='127.0.0.1'
export AGENTROUTER_PORT='7187'
export AGENTROUTER_MODEL_ID='gpt-5.6-sol'
export AGENTROUTER_OPENAI_UPSTREAM='auto'

python proxy.py
```

健康检查：

```bash
curl http://127.0.0.1:7187/health
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AGENTROUTER_API_KEY` | （空） | 上游 API Key；也可写入密钥文件 |
| `AGENTROUTER_BASE_URL` | `https://agentrouter.org` | Anthropic / 默认上游根地址（不要带 `/v1`） |
| `AGENTROUTER_OPENAI_BASE_URL` | 同 `AGENTROUTER_BASE_URL` | OpenAI 网关根地址 |
| `AGENTROUTER_HOST` | `127.0.0.1` | 监听地址 |
| `AGENTROUTER_PORT` | `7187` | 监听端口 |
| `AGENTROUTER_MODEL_ID` | `gpt-5.6-sol` | `/v1/models` 返回的模型 id |
| `AGENTROUTER_OPENAI_UPSTREAM` | `auto` | `auto` / `openai` / `anthropic` |
| `AGENTROUTER_OPENAI_USER_AGENT` | `opencode/1.18.25` | 直连 OpenAI 时的 User-Agent |
| `AGENTROUTER_KEEP_CACHE_CONTROL` | `0` | `1` 时保留块级 `cache_control` |
| `AGENTROUTER_ALLOW_UNKNOWN_FIELDS` | `0` | `1` 时放行更多请求字段 |
| `AGENTROUTER_CONNECT_TIMEOUT` | `15` | 连接超时（秒） |
| `AGENTROUTER_READ_TIMEOUT` | `900` | 读超时（秒，需覆盖长推理） |
| `AGENTROUTER_WRITE_TIMEOUT` | `30` | 写超时（秒） |
| `AGENTROUTER_POOL_TIMEOUT` | `10` | 连接池超时（秒） |
| `AGENTROUTER_MAX_CONNECTIONS` | `50` | 最大连接数 |
| `AGENTROUTER_MAX_KEEPALIVE` | `10` | 最大 keepalive |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `UVICORN_LOG_LEVEL` | `warning` | uvicorn 日志级别 |

密钥文件（环境变量未设置时回退）：

```text
~/.config/opencode/api_keys/AGENT_ROUTER_API_KEY
```

---

## OpenCode 配置示例

### 方式一：Anthropic 协议

```json
{
  "provider": {
    "local-agentrouter": {
      "npm": "@ai-sdk/anthropic",
      "name": "AgentRouter Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:7187",
        "apiKey": "sk-你的密钥"
      },
      "models": {
        "gpt-5.6-sol": {
          "name": "GPT-5.6 Sol (via AgentRouter)"
        }
      }
    }
  }
}
```

注意：`baseURL` **不要**加 `/v1`。AI SDK 会自己拼 `/v1/messages`。

### 方式二：OpenAI Compatible 协议

```json
{
  "provider": {
    "local-agentrouter-oai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AgentRouter OpenAI Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:7187/v1",
        "apiKey": "sk-你的密钥"
      },
      "models": {
        "gpt-5.6-sol": {
          "name": "GPT-5.6 Sol (OpenAI path)"
        }
      }
    }
  }
}
```

注意：`baseURL` **需要**带 `/v1`。AI SDK 会请求 `/v1/chat/completions`。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查与上游信息 |
| `GET` | `/v1/models`、`/models` | 模型列表 |
| `POST` | `/v1/messages`、`/messages` | Anthropic Messages API |
| `POST` | `/v1/chat/completions`、`/chat/completions` | OpenAI Chat Completions API |

流式响应均为 SSE（`text/event-stream`）。

---

## 与 OpenCode 过程输出 / Todo 的关系

| 现象 | 代理侧处理 |
|---|---|
| 工具卡片晚出现或不出现 | Anthropic：工具 `content_block_start` 立即下发；OpenAI：先发 `tool_calls[].function.name` |
| Todo 被清空 / schema 校验失败 | 截断 JSON 尽量括号补全；无法安全修复时保留原始字符串，避免静默变成 `{}` |
| 有工具却不执行 | Anthropic `stop_reason=tool_use`；OpenAI `finish_reason=tool_calls` |
| 流中途报错挂死 | 已下发内容时补齐结束事件，避免客户端一直等待 |
| 思考过程看不到 | Anthropic thinking 块透传；OpenAI 支持 `reasoning_content` 与 `<think>` 拆分 |

---

## 架构示意

```text
OpenCode
  │
  ├─ @ai-sdk/anthropic ──────────────► /v1/messages
  │                                         │
  │                                         ▼
  │                              StreamRepairState（Anthropic SSE）
  │                                         │
  │                                         ▼
  │                              Anthropic Python SDK ──► agentrouter.org
  │
  └─ @ai-sdk/openai-compatible ─────► /v1/chat/completions
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
            OpenAI 直连（httpx2 + UA）                    Anthropic 桥接
            OpenAIRepairState                            OAI → Anthropic 请求
                     │                                   Anthropic SSE 修复
                     │                                   AnthropicToOpenAI 映射
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                   chat.completion.chunk + [DONE]
```

---

## 故障排查

| 问题 | 建议 |
|---|---|
| `unauthorized client detected` | 确认走的是本代理；OpenAI 路径可设 `AGENTROUTER_OPENAI_UPSTREAM=anthropic` 强制桥接 |
| `Connection refused` | 确认 `python proxy.py` 已启动，端口与 OpenCode baseURL 一致 |
| Todo / 工具不执行 | 看代理日志是否出现 tool 相关事件；确认模型支持 function calling |
| 长请求超时 | 增大 `AGENTROUTER_READ_TIMEOUT`（默认 900 秒） |
| 上游 400（thinking / betas） | 保持默认剥离逻辑；不要开 `AGENTROUTER_KEEP_CACHE_CONTROL` 除非你确定上游支持 |

---

## 参考项目

- [agentrouter-spoof-proxy](https://github.com/trefeon/agentrouter-spoof-proxy) — 多客户端头伪装与 OpenAI/Anthropic 反代
- [opencode-agentrouter-support](https://github.com/Fares-Nosair/opencode-agentrouter-support) — OpenCode 侧 AgentRouter 对接思路
- [agentrouter-opencode-proxy](https://github.com/Goodnessmbakara/agentrouter-opencode-proxy) — 用 Python SDK 绕过 WAF 的说明

---

#### agentrouter注册地址：https://agentrouter.org/register?aff=qJrT

## 许可

按你的仓库需要自行补充 License。本 README 仅描述 `proxy.py` 的用途与配置方式。

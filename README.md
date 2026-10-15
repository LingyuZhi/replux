# Replux

Replux 是一个终端 AI 助手。支持通过 Anthropic Messages 或 OpenAI Responses 完成多轮流式问答。
回复正文会随生成过程显示。后续提问会携带此前完整成功的交流，退出后不保存对话。

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv sync --dev
uv run replux
```

启动前创建 `~/.replux/config.yaml`，填写完整模型名与密钥：

```yaml
protocol: anthropic
model: your-full-model-id
base_url: https://api.anthropic.com
api_key: your-api-key
max_output_tokens: 8192
timeout_seconds: 120
```

使用 OpenAI Responses 时，将配置改为：

```yaml
protocol: openai
model: your-full-openai-model-id
base_url: https://api.openai.com/v1
api_key: your-api-key
```

OpenAI 请求使用 `store: false`。OpenAI 配置不接受 Anthropic 专用的 `thinking` 字段。

也可以指定另一份配置，或通过 Python 模块启动：

```sh
uv run replux --config /path/to/config.yaml
uv run python -m replux --config /path/to/config.yaml
```

每次启动仅读取一份配置，修改后需重启。`api_key` 按字面读取；
`base_url` 直接交给 SDK，兼容端点需填写其要求的根地址。
最后两个字段可省略。`timeout_seconds` 控制网络 I/O 等待时间。
请求自动重试已关闭。

在底部输入问题，按 `Enter` 发送。空闲时按 `Ctrl+C` 退出。
未完成或截断的回复会保留在界面并标记，不进入后续上下文；需要重发问题。
网络或 API 请求失败仍会显示提示并退出。完整请求错误恢复将在后续版本加入。

开发检查：

```sh
uv run python -m pytest -q
uv run mypy
```

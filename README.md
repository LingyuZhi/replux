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

Anthropic 可按模型支持情况添加 `thinking`。省略时保持模型默认行为。
例如，显式使用 adaptive 并请求可展示摘要：

```yaml
thinking:
  type: adaptive
  display: summarized
```

手动模式可写为 `thinking: {type: enabled, budget_tokens: 1024}`。
预算必须至少为 `1024`，且小于 `max_output_tokens`；也可添加 `display: summarized`。
显式关闭使用 `thinking: {type: disabled}`，不能携带预算或展示参数。
模型拒绝配置时会报错，不自动更换模式或重试。

API 返回可展示思考内容后，每轮显示默认折叠的 `+ Thinking`。
点击 `+` 展开、点击 `-` 收起；用 `Tab` 移动焦点后，也可按 `Enter` 切换。
展开后继续显示新增内容。显示操作不改变请求；没有可展示内容时不出现控件。

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
`max_output_tokens` 和 `timeout_seconds` 可省略。`timeout_seconds` 控制网络 I/O 等待时间。
请求自动重试已关闭。

在底部输入问题，按 `Enter` 发送，按 `Ctrl+J` 换行。可粘贴中文多行内容，保留换行与缩进；纯空白不会发送。
回复按纯文本显示。停留在底部时跟随新内容；向上滚动查看旧记录时保持位置，回到底部后恢复跟随。
可用 `Tab` 切换焦点，在对话区用 `PageUp` / `PageDown` 翻页、`End` 回到底部。
生成时按 `Ctrl+C` 取消当前回复，等待请求流关闭后可再次发送；
空闲时按 `Ctrl+C` 退出。生成期间可以编辑下一条草稿，但不能重复发送。
取消、未完成或截断的回复会保留在界面并标记，不进入后续上下文；需要重发问题。
认证、限流、网络或 API 错误会显示在当前轮次，保留草稿并恢复输入，不自动重试。
启动配置错误和程序缺陷仍会报错退出。

开发检查：

```sh
uv run python -m pytest -q
uv run mypy
```

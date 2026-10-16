# Replux

Replux 是一个终端 AI 助手。支持通过 Anthropic Messages 或 OpenAI Responses 完成多轮流式问答。
回复正文会随生成过程显示。后续提问会携带此前完整成功的交流，退出后不保存对话。

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv sync --dev
```

创建 `~/.replux/config.yaml`（目录不存在时先创建 `~/.replux/`），填写完整模型名与密钥。
`providers` 中只启用一项，其余配置整块注释。`name` 是便于识别的名称；
`protocol` 选择 `anthropic`（Messages）或 `openai`（Responses）：

```yaml
providers:
  - name: anthropic-official
    protocol: anthropic
    base_url: https://api.anthropic.com
    api_key: your-api-key
    model: claude-opus-5-5
    thinking: auto

  # - name: openai-official
  #   protocol: openai
  #   base_url: https://api.openai.com/v1
  #   api_key: your-openai-api-key
  #   model: your-full-openai-model-id
  #   thinking: auto
```

保存配置后启动：

```sh
uv run replux
```

`thinking` 可选 `auto`、`low`、`medium`、`high`、`xhigh` 或 `max`。
`auto`（也可省略）沿用服务端默认行为；其他值只设置推理强度，不指定 token 预算。
不同模型支持的档位与默认值可能不同。模型拒绝配置时会报错，不自动更换档位或重试。
思考模式和可见摘要遵循服务端默认设置；配置强度不保证返回可展示的思考内容。

API 返回可展示思考内容后，每轮显示默认折叠的 `+ Thinking`。
点击 `+` 展开、点击 `-` 收起；用 `Tab` 移动焦点后，也可按 `Enter` 切换。
展开后继续显示新增内容。显示操作不改变请求；没有可展示内容时不出现控件。

使用另一项配置时，先注释当前项，再取消另一项的注释。OpenAI 请求使用 `store: false`。

也可以指定另一份配置，或通过 Python 模块启动：

```sh
uv run replux --config /path/to/config.yaml
uv run python -m replux --config /path/to/config.yaml
```

每次启动仅读取一份配置，修改后需重启。`api_key` 按字面读取；
`base_url` 直接交给 SDK，兼容端点需填写其要求的根地址。
输出与超时由应用处理：Anthropic 的单次输出上限为 `128000` tokens，包含思考与正文；
这是面向前沿模型的应用上限，模型必须支持此值。OpenAI 沿用服务端输出默认值。
网络 I/O 等待采用 SDK 默认值：连接为 5 秒，其余为 600 秒，不是整轮生成的总时限。
请求自动重试已关闭。旧版平铺配置仍可读取。

在底部输入问题，按 `Enter` 发送，按 `Ctrl+J` 换行。可粘贴中文多行内容，保留换行与缩进；纯空白不会发送。
回复按纯文本显示。停留在底部时跟随新内容；向上滚动查看旧记录时保持位置，回到底部后恢复跟随。
可用 `Tab` 切换焦点，在对话区用 `PageUp` / `PageDown` 翻页、`End` 回到底部。
生成时按 `Ctrl+C` 取消当前回复，等待请求流关闭后可再次发送；
空闲时按 `Ctrl+C` 退出。生成期间可以编辑下一条草稿，但不能重复发送。
取消、失败或截断的轮次会保留在界面并标记，问题和半截回复均不进入后续上下文；
需要重新提交完整问题，不能仅用“继续”续接该轮。
认证、限流、网络或 API 错误会显示在当前轮次，保留草稿并恢复输入，不自动重试。
启动配置错误和程序缺陷仍会报错退出。

开发检查：

```sh
uv run python -m pytest -q
uv run mypy
```

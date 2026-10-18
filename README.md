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

界面记录中的用户输入使用灰色背景和 `›` 标记。多行输入与 Reply 的续行保持正文对齐；
Reply 首行使用明亮圆点，空 Reply 不占显示空间。Reply 随生成过程排版为 Markdown，
支持标题、强调、列表、代码高亮、引用和表格；工具前后的 Reply 各自排版。
用户输入、Thinking 和工具详情保留原有显示方式。
长代码行和宽表格在各自区域内横向滚动，溢出时显示细滚动条。
用 `Tab` / `Shift+Tab` 进入或离开溢出区域，按 `Left` / `Right` 横向阅读；未溢出区域不增加焦点停留。

API 返回非空白的可展示思考内容后，在本轮输入下方显示一行默认折叠的 `+ Thinking`。
点击 `+` 展开、点击 `-` 收起；用 `Tab` 移动焦点后，也可按 `Enter` 切换。
展开后继续显示新增内容，各轮独立保留展开状态。正常、失败或取消的轮次实际结束后，
标题改为 `Thoughts`。显示操作不改变请求、上下文或草稿；没有可展示内容时不出现控件。

每次工具调用独立显示动作、目标和一行结果摘要，默认折叠。点击标题，或用 `Tab` 聚焦后按
`Enter`，可查看工具名、原始参数、状态、可用耗时及完整受限结果，再次操作可收起。
摘要只使用真实返回的统计；`Truncated` 表示结果有截断，详情保留日志路径和副作用说明。
详情不会读取日志或再次执行工具。`Failed`、`Cancelled` 和 `Skipped` 分别表示失败、取消和未执行；
后续 Reply 正常完成也不会覆盖工具失败。命令日志随当前会话保留，退出后清理。

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
输入区使用上分隔线和 `›`，随内容扩高。运行时，上方显示当前动作、本地总经过时间和取消提示。
`Working…` 表示等待具体活动，`Thinking…` 表示收到思考内容，`Writing reply…` 表示生成正文；
工具运行时显示动作与目标。活动动画集中在底部，工具状态仍可查看。
每轮实际结束后留下灰色耗时记录：`Finished in`、`Cancelled after` 或 `Failed after`。
`Finished` 表示 Turn 流程完成，工具失败仍单独保留。
点击 Reply 中的 http(s) 链接可打开默认浏览器；本地路径仅显示，不提供文件跳转。
停留在底部时跟随新内容；向上滚动查看旧记录时，Markdown 重排会保留正在阅读的内容位置，回到底部后恢复跟随。
可用 `Tab` 切换焦点，在对话区用 `PageUp` / `PageDown` 翻页、`End` 回到底部。
生成时按 `Ctrl+C` 取消当前 Turn，清理期间持续显示 `Cancelling…`，实际清理结束后可再次发送；
空闲时按 `Ctrl+C` 退出。生成期间可以编辑下一条草稿，但不能重复发送。
清理期间也可编辑草稿；重复取消不会跳过清理。
取消、失败或截断的轮次会保留在界面并标记。未开始执行工具时，问题和半截回复不进入后续上下文，
需要重新提交完整问题。已经开始执行工具时，保留原始调用及已知结果，不承诺回滚已发生的副作用；
最终 Reply 仍仅在完整成功后进入后续上下文。
认证、限流、网络或 API 错误会显示在当前轮次，保留草稿并恢复输入，不自动重试。
启动配置错误和程序缺陷仍会报错退出。

开发检查：

```sh
uv run python -m pytest -q
uv run mypy
```

# Replux

Replux 是一个终端 AI 助手。目前支持通过 Anthropic Messages 完成单轮流式问答。
每次提问独立发送，回复正文会随生成过程显示。

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
请求失败会显示提示并退出。完整请求错误恢复将在后续版本加入。

开发检查：

```sh
uv run python -m pytest -q
uv run mypy
```

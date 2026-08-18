# Coding agents (VS Code Copilot, OpenClaw)

NoLlama can drive tool-calling coding agents — the model emits function calls,
NoLlama parses them into OpenAI/Ollama `tool_calls`, and the agent acts on the
results.

![OpenClaw running locally against NoLlama on an Intel iGPU — start-openclaw.ps1 brings up NoLlama (GPU, Qwen2.5-Coder-7B), pre-warms the cache, and the agent replies](screenshots/openclaw-1-Skjermbilde2026-06-28_113203.png)
*OpenClaw driving a local Qwen2.5-Coder model on an Intel iGPU via NoLlama — one command (`./start-openclaw.ps1`), no cloud, no NVIDIA.*

> **Tool calling runs on GPU/iGPU and CPU — not the NPU.** The NPU has a hard
> prompt cap and small NPU-class models can't reliably drive agent loops, so
> NoLlama ignores `tools` there and answers as plain chat; `/api/show` advertises
> the `tools` capability only for GPU/CPU slots. Load a coder LLM on the GPU, or
> on a strong desktop CPU (many-core Core Ultra) where prefill can beat a weak
> iGPU. The Qwen2.5-Coder GPU builds in the menu work well; pick a smaller size
> (7B) for snappier prefill on big agent prompts.
>
> Tool turns are **buffered** (the whole reply is generated before the structured
> `tool_calls` are sent), but the server emits SSE keep-alive pings during a long
> prefill so agent clients (Copilot/OpenClaw) don't hit their idle timeout and
> abort. Big agent system prompts (~20k tokens) prefill slowly on weak iGPUs — a
> smaller model, the CPU, or trimming the client's tool set all help. And
> **prefix caching is on by default**, so that big system prompt is prefilled
> once, not every turn — after the first turn, agent turns are fast (~47x on the
> cached prefix). Disable with `--no-prompt-cache`.

The tool prompt is rendered in Qwen3-Coder native format, and `parse_tool_calls`
also understands Hermes, Mistral `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek,
and bare-JSON outputs — so most instruct/coder models work.

**VS Code Copilot Chat** (0.53+) — point it at the Ollama API and start the
server with `--vscode-compat` so VS Code accepts the version handshake:

```powershell
python nollama.py --gpu-model-dir gpu-coder-model --vscode-compat
```

Then in VS Code set the Ollama base URL to `http://localhost:11434` and pick the
GPU model. (Add `--debug` while wiring it up to see exactly what Copilot sends.)

**OpenClaw** — speaks the OpenAI chat-completions API NoLlama already serves; it
runs against a NoLlama GPU slot with no code changes, just config. See
[OPENCLAW-PLAN.md](OPENCLAW-PLAN.md) for the step-by-step setup (the one gotcha:
address the model as `<name>@GPU` so tool requests hit the GPU, not the NPU).

**Install OpenClaw** (once):

```powershell
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

Then **`start-openclaw.ps1`** is the one-command launcher (the NoLlama equivalent
of `ollama launch openclaw`):

```powershell
./start-openclaw.ps1 -Setup -Device GPU     # -Setup writes the `nollama` provider into openclaw.json
./start-openclaw.ps1 -Device GPU            # subsequent runs
```

It starts NoLlama with the agent flags (`--device`, `--prewarm`, keep-loaded),
waits until ready, then runs OpenClaw. If a NoLlama is **already** on the port it
**verifies** it (prefix caching on + a tool-capable GPU/CPU slot) and reuses it —
or, if it's misconfigured, tells you why and offers to restart it correctly
(`-Force` to skip the prompt). `-Warmup` fires one throwaway turn first so even
the first real turn is fast.

> **NoLlama runs OpenClaw in a deliberately constrained mode — by design.** A
> coding-agent prompt is large (~21k tokens of system prompt + tool schemas), which
> is a lot for a small local model on weak Intel hardware. So `-Setup` doesn't just
> point OpenClaw at NoLlama — it also **trims OpenClaw** to fit: it selects the
> `coding` tool profile and turns off web search, X search, memory search, and the
> startup-context prelude. This shrinks the prompt and tool surface so a 7B coder on
> an iGPU/CPU can actually drive the loop. It's all plain config in
> `~/.openclaw/openclaw.json` — re-enable anything if your hardware can handle a
> bigger prompt, and re-run `-Setup` to restore the trimmed defaults. Package
> updates (`npm i -g openclaw@latest`) don't touch this config; only re-running
> `openclaw onboard` might, in which case re-run `-Setup`.

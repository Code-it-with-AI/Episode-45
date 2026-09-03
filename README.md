# Episode-45: Qwen 3.8 on NInfer
After testing multiple runtimes for hosting Qwen, Carl finally settles on NInfer.

📺 YouTube video: https://youtu.be/

🏠 Code it with AI Home Page: https://codeitwithai.com

# The Search for a Stable Local Coding LLM

## Background

The goal is to run a capable private local coding model for GitHub Copilot CLI on Windows 11 with an NVIDIA RTX 5090 (32 GB VRAM), with a large context window, reliable tool calling, and good long-session stability.

Qwen3.8-27B became the preferred model because its coding/debugging quality was strong. The harder problem has been finding an inference framework that combines that quality with 128K+ context and dependable performance.

## 1. llama.cpp --- Great Until It "Bonked"

The original setup used llama.cpp on Windows with Qwen3.8-27B GGUF, 128K context, Flash Attention, Jinja tool/chat templates, one concurrent context slot, and an OpenAI-compatible LAN server.

When it worked, it worked very well. The recurring problem became known as the **llama.cpp bonk**: after substantial Copilot use, inference would effectively stop making useful progress for many minutes. Restarting llama.cpp immediately restored good performance.

Cache-related changes and `GGML_CUDA_DISABLE_GRAPHS` did not solve it. llama.cpp was updated to build 10730 during troubleshooting, but the behavior persisted.

**Bottom line:** excellent 128K performance and useful Copilot behavior, but unreliable over long sessions.

## 2. vLLM --- Stable and Fast, but Not Enough Context

vLLM was tested under WSL2. Getting it running required Linux build tools, CUDA toolkit/NVCC configuration, FlashInfer installation and version alignment, JIT cache setup, and several Python dependency fixes.

Eventually vLLM 0.28.0 with PyTorch 2.13.0+cu130 successfully served `Inferact/Qwen3.8-27B-NVFP4`. The OpenAI-compatible API worked with Copilot CLI.

The limiting factor was GPU memory for KV cache. Practical context topped out around the mid-50K range; a 56K configuration worked, but Copilot eventually exceeded it and vLLM correctly returned an HTTP 400 context-limit error.

Later research found newer 4-bit TurboQuant KV-cache work that could potentially allow much larger contexts, but the installed vLLM did not expose TurboQuant and current upstream reports indicated problems with TurboQuant on hybrid GDN models such as Qwen3.8. That experiment was deferred.

**Bottom line:** probably the cleanest/stablest server tested, but the tested FP8 KV configuration could not provide enough context.

## 3. SGLang --- Promising Until It Destabilized WSL

SGLang successfully loaded Qwen3.8-27B NVFP4, recognized its hybrid GDN/Mamba architecture, and allocated Mamba and FP8 KV caches.

Startup first failed during FlashInfer autotuning. Adding `--disable-flashinfer-autotune` allowed it to proceed.

The next problem was CUDA graph capture. With CUDA graphs enabled, SGLang reached `Capture target prefill CUDA graph begin`, after which WSL became unresponsive. New WSL sessions failed with `Wsl/Service/0x8007274c` until `wsl --shutdown` was used.

Disabling both FlashInfer autotuning and CUDA graphs produced a working server, but it was noticeably too slow. Re-enabling CUDA graphs reproduced the WSL instability.

**Bottom line:** conservative mode worked but was slow; fast mode repeatedly destabilized WSL.

## 4. Ollama --- 128K Fits, but Copilot Sessions Become Pathological

Ollama 0.33.2 was revisited after discovering that the previous 32K context had been an automatic default rather than a hard limit.

With `OLLAMA_CONTEXT_LENGTH=131072`, the model `qwen3.8:27b-q4_K_M` loaded successfully. `ollama ps` confirmed:

-   **131,072-token context**
-   **100% GPU**
-   about **25 GB loaded**

This was extremely encouraging. Copilot initially ran quickly.

As sessions grew, however, Copilot began spending extraordinary amounts of time at `Thinking...` and `Compacting conversation history...`. GPU monitoring showed that the RTX 5090 was not idle: it remained around 96--97% utilization and roughly 500 W during some of these periods.

One Copilot context report showed about 45K tokens during compaction. Another later showed 109K/139K tokens, including 89.6K tokens of messages.

Starting a completely new Copilot session restored good speed without restarting Ollama, indicating that accumulated session state was a major factor. But the workflow was still unacceptable: only a handful of user commands could cause enormous context growth and very long reasoning periods. A simple "Enumerate Devices button does nothing" report expanded into multiple file reads, application launching, endpoint testing, and a very long reasoning episode.

Narrow, explicit prompts helped, but requiring constant micromanagement was not considered a satisfactory coding workflow.

**Bottom line:** technically impressive---Qwen3.8 at 128K entirely on the 5090---but Copilot/Qwen session growth and reasoning behavior made interactive use unreliable.

## 5. Comparison So Far

| Framework | Best Result | Main Problem |
|---|---|---|
| llama.cpp | 128K, fast, useful | Eventually "bonks"; restart restores it |
| vLLM | Stable and fast | ~50–60K practical context in tested FP8 setup |
| SGLang | Qwen3.8 loads and runs | CUDA graphs wedge WSL; safe mode too slow |
| Ollama | 128K, 100% GPU | Copilot sessions balloon and become painfully slow |
| **NInfer** | **262K text / 128K vision, fast and stable so far** | Copilot reports a lower ~139K context budget; persistent Copilot instructions can be lost during compaction |

## 6. NInfer --- The First Clear Winner

The next experiment used **NInfer**, specifically the native Windows RTX 5090 build. Unlike the previous Python/CUDA servers, this avoided WSL entirely and was purpose-built for the hardware and model being tested.

Selected text package:

`ninfer-windows-v1.0.3-rtx5090.zip`

Model:

`qwen3_8_27b_nvfp4.ninfer`

The model download was approximately 20 GB. The supplied RTX 5090 launch profile was configured for:

- Qwen3.8-27B NVFP4
- native Windows
- OpenAI/Anthropic-compatible serving on port 8080
- `--max-context 262144`
- `--kv-capacity 262144`
- FP8 KV cache
- one concurrent request
- MTP speculative decoding with five draft tokens
- LM-head drafting
- preserved Qwen thinking
- 16 GB host KV cache (`--host-kv-mib 16384`)
- a 10-minute pending-request timeout

For LAN access from the development machine, the supplied `--host 127.0.0.1` setting was changed to `--host 0.0.0.0`.

### Port 8080 cleanup

The first NInfer launch failed with:

`failed to bind 0.0.0.0:8080`

Windows showed `svchost` already listening on port 8080. This was the old `netsh interface portproxy` rule that had been created earlier to forward Windows port 8080 into WSL for vLLM. Because NInfer runs natively on Windows, that forwarding rule was no longer required. Removing the old portproxy freed port 8080 while retaining the Windows firewall rule needed for LAN access.

### Successful 262K startup

After the port conflict was removed, NInfer loaded successfully. Its startup log reported:

- model weights: about **19.73 GiB**
- model load time: about **8.4 seconds**
- explicit KV capacity: **262,144 tokens**
- KV/runtime allocation: about **9.11 GiB**
- approximately **1.28 GiB GPU memory free after startup**
- **16 GiB host KV cache**
- server listening on `0.0.0.0:8080`
- model ID `qwen3.8-27b`

This was the first tested server to actually start Qwen3.8-27B on the single RTX 5090 with a **262K configured context capacity**.

### Copilot CLI results

GitHub Copilot CLI connected successfully through the OpenAI-compatible endpoint and immediately produced fast responses. More importantly, it continued to be fast and useful during real coding work rather than only simple test prompts.

The initial impression after sustained use was significantly better than the previous frameworks: **snappy, proactive, and able to finish real jobs**. After recreating a project that an earlier agent had constructed incorrectly and giving Qwen clearer development instructions, its coding behavior improved substantially. It was then able to take a somewhat higher-level prompt and work through the task successfully without the constant micromanagement that had been necessary during the Ollama test.

After several hours of real development work, NInfer remained responsive and productive. At that point it became the first framework in the experiment that felt like a practical daily-driver candidate rather than merely a promising test.

## 7. NInfer Vision Profile

The text-only NInfer server correctly rejected image-bearing Copilot requests with:

`400 Vision is disabled for this server`

An important Copilot behavior was discovered at the same time: once a screenshot had been attached to a conversation, Copilot continued to include that image in subsequent requests. As a result, even later text-only prompts were rejected by the text-only NInfer server.

The session could be rescued without abandoning its work by using Copilot's `/undo` or `/rewind` function, rewinding the **conversation only** to the point immediately before the screenshot while leaving the project files unchanged.

The separate NInfer vision package was therefore installed for normal use when screenshots are needed:

`ninfer-windows-v1.0.3-rtx5090-vision.zip`

The supplied vision profile trades context capacity for the additional VRAM required by vision. Its configured context is **131,072 tokens (128K)** rather than the text profile's 262,144 tokens. The same Qwen3.8-27B NVFP4 model file can be used, so another ~20 GB model download is not necessary.

The vision build also proved fast and effective in Copilot. This makes it a strong everyday option when screenshot input is important, while the 262K text profile remains attractive for maximum-context coding sessions.

## 8. Copilot Context Accounting and Compaction

A remaining issue is **Copilot CLI's own context accounting**.

Even when connected to the NInfer text server configured for 262,144 tokens, Copilot displayed a model budget of approximately **139K**. For example, a fresh session showed roughly `20k/139k`, even though NInfer had explicitly allocated a 262K KV capacity.

As sessions grew, Copilot began automatic conversation compaction near its perceived limit. Examples observed included:

- about **108K/139K (78%)**, followed by compaction - about **137K/139K (99%)**, with responsiveness beginning to fall - in one anomalous case, Copilot's logical history accounting reached **358K/139K (257%)**, with negative reported free space

The 358K figure should not be interpreted as a single 358K request being accepted by NInfer; it was Copilot's logical history accounting. NInfer's configured active context limit remained 262K.

Restarting Copilot when it approaches its own context-management limit is currently a practical way to restore a clean, responsive session without restarting NInfer.

A future task is to determine whether Copilot CLI's custom-provider/model metadata can be changed so that it recognizes NInfer's actual context capacity rather than using the approximately 139K budget.

## 9. Persistent Agent Instructions

Another Copilot-side issue emerged during long sessions: behavioral rules originally supplied through `startup.md` were sometimes forgotten, especially after conversation compaction. Two examples were:

- play `ahem.wav` to get the user's attention
- provide a Git commit message when a requested task is complete

Project-state files such as `AI_MEMORY.md` and `AI_CURRENT.md` continued to be useful for checkpointing work between Copilot sessions, but mandatory behavioral rules should ideally live in Copilot's persistent repository/custom-instruction mechanism rather than depending only on a file read early in the conversation. That should make the rules less vulnerable to being summarized away during compaction.

## 10. Lessons From the Experiment

Several conclusions became clear during this search:

1. **Advertised model context and usable agent context are different things.** A server can technically expose a huge context while the client applies its own smaller context budget or compaction policy.
2. **A fast first response proves very little.** The meaningful test is hours of real agentic coding with file reads, shell commands, tool calls, builds, tests, and accumulated history.
3. **Qwen3.8 benefits from a correctly structured project and clear development rules.** Some apparently poor model behavior was actually triggered by inheriting a project that an earlier agent had created incorrectly.
4. **WSL added substantial failure surface.** vLLM and especially SGLang required significant CUDA/Python dependency work, and SGLang could destabilize WSL itself. Native Windows NInfer eliminated that entire layer.
5. **The inference engine matters enormously.** The same basic Qwen3.8 model family behaved very differently across llama.cpp, Ollama, vLLM, SGLang, and NInfer.
6. **NInfer's specialization is an advantage here.** Rather than being a general-purpose inference framework, the tested Windows build is narrowly optimized for the RTX 5090/Qwen3.8 combination, which is exactly the target hardware and workload.

## Current Baseline

As of the latest test, the preferred local coding setup is:

- **GPU:** NVIDIA GeForce RTX 5090, 32 GB VRAM
- **Operating system:** Windows 11
- **Coding client:** GitHub Copilot CLI
- **Inference server:** **NInfer Windows RTX 5090 build v1.0.3**
- **Model:** **Qwen3.8-27B NVFP4**
- **Text profile:** **262,144-token configured context**, FP8 KV, MTP5
- **Vision profile:** **131,072-token configured context**, vision enabled
- **API:** OpenAI-compatible endpoint on port 8080
- **LAN server address:** `192.168.1.155:8080`

After a couple of hours of sustained real coding, the NInfer setup was described as **snappy, proactive, and able to finish the job**. It then continued successfully on additional, higher-level development work.

**Current verdict: NInfer is the winner of the experiment so far.**

------------------------------------------------------------------------

*Experiment log started September 2026. Updated after the successful NInfer text and vision tests and several hours of sustained Copilot CLI coding.*

## Resource Links

- **About Rocky Lhotka:** https://about.me/rockfordlhotka


# Testing NoLlama in Docker — protocol

For [issue #31](https://github.com/aweussom/NoLlama/issues/31). Answer it
with measurements. Docker/WSL install is assumed done.

**Out of scope, decided:** the NPU. It is not exposed to WSL2 or Docker
Desktop Linux containers ([WSL #40842](https://github.com/microsoft/WSL/issues/40842),
[Intel Community](https://community.intel.com/t5/Graphics/NPU-in-WSL/td-p/1581143));
Anything we publish says "no NPU" in plain words.

**Correction (2026-08-24): there is no "WSL 3".** Microsoft
[denied it](https://www.windowslatest.com/2026/06/28/microsoft-denies-wsl-3-exists-reveals-windows-11s-wsl-containers-ship-next-week/)
- what Build 2026 announced is **WSL Containers**, built on WSL 2, and much
of the press mislabelled it. It is in public preview via
`wsl --update --pre-release` (WSL 2.9.3+), GA targeted for autumn 2026
([devblog](https://devblogs.microsoft.com/commandline/wsl-container-is-now-available-for-public-preview/)).
Whether it changes the NPU answer is **unverified** - that claim came from
the same mislabelled reporting, so treat it as rumour until someone runs
`available_devices` inside it.

## Ground rules

**No model downloads into a container. Ever.** Pulling 4-20 GB into a
throwaway is wasteful and pointless when the host already has the weights.
This is not a preference, it is a design constraint on everything below:

- Models are **bind-mounted read-only** from the host's models directory.
- The image is **model-free**. Nothing baked in, nothing fetched at build or
  run time. (Also the licence-clean choice: baking Gemma weights into an
  image is a redistribution question under the Gemma Terms. Avoided entirely
  by never doing it.)
- If a test seems to need a model we do not have, the test is wrong.

That makes bind-mounting mandatory rather than incidental, which promotes
hazards **1.1** (mount path vs the naming rule) and **1.4** (integrity check
over drvfs) from "worth checking" to "on the critical path".

### What is already on disk, and what each is for

| Model | Size | Use in this protocol |
|---|---|---|
| `SmolLM3-3B-int4-cw-ov` | 1.6 GB | **Default.** Plumbing, ports, streaming, restarts - smallest thing that proves the path |
| `gemma-4-E2B-it-int4-ov` | 4.1 GB | VLM path + the probe set; prefix-cache cold-vs-repeat |
| `Qwen2.5-VL-3B-Instruct-int8-ov` | 3.9 GB | Second VLM opinion if a result looks odd |
| `gemma-4-E4B-it-int8-ours` | 7.8 GB | Parity against the 1.16 s steady-state TTFT already measured |
| `gemma-4-26b-a4b-it-int4-ov` | 15 GB | **Only** for hazard 1.3 - the WSL2 memory-cap question needs a big model to be a question at all |

Iterate on SmolLM3. Touch the 26B once, deliberately, for 1.3.

Copying **one** small model onto the WSL2 ext4 filesystem is allowed for the
1.4 drvfs-vs-native timing comparison - that is a local copy, not a download.

---

## How bleeding-edge are we willing to be?

Standing rule, unchanged: **test on nightly, ship on release.** That is what
`venv-nightly` and `install.ps1 -Nightly` exist for, and it is the same rule
that pulled Qwen3.8 27B out of `models.json` on 2026-08-21. Testing here on a
preview stack is fine; a *supported* container path is not announced until
every layer below it is on a release.

| Layer | Needed for | Status | Verdict |
|---|---|---|---|
| WSL 2 | GPU via `/dev/dxg` | shipped | fine |
| Intel compute-runtime | GPU inside the distro/container | current **release** on GitHub; Ubuntu's packaged build lags for Battlemage | fine — an upstream release, just not the distro's |
| OpenVINO | the runtime itself | 2026.3 release works natively; nightly image available if the container needs newer | test on nightly, ship on release |
| WSL Containers | nothing we need | public preview, GA autumn 2026 | **skip** - see below |

**We do not want WSL Containers.** It is a container runtime inside WSL 2 for
running Linux containers on Windows without Docker Desktop - and every
benefit it offers is already covered by running plain `docker-ce` inside the
WSL 2 distro, which also dodges the Docker Desktop licence.

The deciding argument is portability. #31 asks for **Linux** deployment: a
`Dockerfile` and `compose.yml` built against ordinary `docker-ce` run on the
requester's hardware. Anything WSL-Containers-specific is Windows-only and
does not serve the request that prompted this work.

The one thing that could change that is its rumoured NPU passthrough - but
that claim comes from the same mislabelled "WSL 3" reporting, and even if it
holds it would be a Windows-only NPU path: nice for local dev, irrelevant to
#31, and NPU is scoped out anyway. Re-check only if someone demonstrates
`available_devices` listing NPU inside it.

So the realistic target for this experiment is **GPU and CPU only**, which is
also exactly what #31 can honestly be offered.

---

Run the phases in order and stop at the first hard failure — each one gates
the next, and an early stop is a valid answer for #31.

---

## Phase 0 — gating: is the GPU visible *and correct* inside a container?

Visibility is not usability. This project has already been burned by a device
that enumerated fine and then computed garbage (openvino#37419), so check
both.

```bash
docker run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  openvino/ubuntu24_runtime:latest \
  python3 -c "import openvino as ov; c=ov.Core(); print([(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

| Result | Next |
|---|---|
| lists `GPU` = "Arc Pro B60" | Phase 1 |
| `CPU` only | **Stop.** Close #31 with the evidence; Linux-native may still work, note it as untested-here |

On native Linux the equivalent is `--device /dev/dri/renderD128`; we cannot
test that here, so any claim about it stays explicitly untested.

---

## Phase 1 — the NoLlama-specific hazards

The interesting failures are not "does the server start". These are the
places where containerization collides with something this codebase actually
does. Each has a concrete test and a concrete expected result.

### 1.1 Model naming — the directory name is authoritative

`resolve_display_name` uses the directory name as the API/UI model id and
only follows a symlink when that name is generic (`model/`, `gpu-model/`).
A bind mount is a rename waiting to happen.

```bash
# right: name preserved
-v /c/Users/wossn/models/gemma-4-E2B-it-int4-ov:/models/gemma-4-E2B-it-int4-ov
# wrong: clients would have to request "model"
-v /c/Users/wossn/models/gemma-4-E2B-it-int4-ov:/models/model
```

**Test:** `--scan` in-container, and `GET /v1/models`. **Expect** the real
directory name, not `model`. If compose encourages the generic form, the
compose file is wrong, not the code.

### 1.2 KV pool sizing under a container memory limit

`_resolve_kv_pool` and `_preflight_memory` size from the *device budget* —
GPU via `GPU_DEVICE_TOTAL_MEM_SIZE`, CPU via total RAM. A container sees host
RAM unless it reads cgroup limits, so the preflight may promise memory the
container cannot have.

```bash
docker run --memory=8g ... nollama --scan          # and a real load
```

**Expect:** the logged KV pool and token capacity to reflect **8 GB**, not
host RAM. If it reports host RAM, that is a real bug worth fixing regardless
of whether we ship Docker support — issue #21's `Got unfinished
GenerationStatus` is what a too-small pool produces.

### 1.3 Model load stages through host RAM

Loading a big model peaks at roughly model-sized host RAM even on a discrete
card (recorded in `NEXT-STEPS.md`). WSL2 caps its VM at ~50% of host RAM by
default — on this 32 GB box that is ~16 GB, and the 26B needs 14.3 GB.

A `.wslconfig` was created 2026-08-24 in the user profile setting
`memory=24GB`, `swap=16GB` and `autoMemoryReclaim=gradual` precisely for
this, so the test measures the model rather than the default ceiling.

**Test:** load `gemma-4-26b-a4b-it-int4-ov` in-container. **Expect** a clean
load. **Also record** what it needs, because anything we ship must state the
`.wslconfig` requirement - a 32 GB box on defaults would thrash.

### 1.4 Weight integrity check over a bind mount

`_verify_weights_integrity` reads offsets from the `.xml` and checks the
`.bin` size. Over `/mnt/c` (drvfs) this is slow.

**Test:** time `--scan` against a bind-mounted model vs the same model on the
WSL2 ext4 filesystem. **Expect** correctness either way; **record** the time
difference, because "models on /mnt/c are slow" is the kind of thing that
looks like a hang.

### 1.5 Both API ports

NoLlama serves OpenAI on 8000 **and** Ollama on 11434. A compose file that
publishes only 8000 silently breaks every Ollama client.

**Test:** `/v1/models` on 8000 and `/api/tags` on 11434 from the host.

### 1.6 Streaming and the SSE heartbeat

Tool turns are buffered and rely on SSE keep-alives every `HEARTBEAT_SECS` to
stop client watchdogs aborting. A proxy or port-publishing layer that buffers
would defeat exactly the mechanism that exists to prevent aborts.

**Test:** a streaming request and a tools request through the published port;
confirm tokens/pings arrive incrementally, not in one lump at the end.

### 1.7 Writable state

`--idle-timeout 0` auto-enables prewarm, which writes `prewarm-<port>.json`
to the working directory. A read-only rootfs or a non-root user without write
access turns that into a startup failure.

**Test:** run with `--idle-timeout 0`, restart, confirm the prewarm file is
written and reused (log shows the startup prefill).

---

## Phase 2 — parity against numbers we already have

Same models, same probes, so the comparison is apples to apples. Native
figures are in `docs/dev/models.md` and `docs/MODELS.md`.

| Measure | Native (B60, 2026.3) | Container |
|---|---|---|
| `gemma-4-E2B-it-int4-ov` load time | ~16 s | ? |
| prefix cache: cold vs repeat TTFT | 3.1 s → 0.5 s | ? |
| `aweussom/gemma-4-E4B-it-int8-ov` steady TTFT | 1.16 s | ? |
| probe set correctness | 7 cases, known answers | must match exactly |

Reuse the probe harness — identical inputs, so any divergence is the
container and not the prompt. **It currently lives in a session scratchpad
and will vanish**; move it into the repo alongside `qa_vlm.py` before relying
on this section.

**A tok/s gap is publishable either way.** "Containers cost X%" is useful;
"containers are free" is more useful.

---

## Phase 3 — packaging, only if Phases 0–2 pass

- `Dockerfile` — OpenVINO runtime base, `requirements.txt`, no model baked in
- `docker-compose.yml` — ports 8000 + 11434, models bind-mounted **under
  their real directory names** (1.1), device + `/usr/lib/wsl` passthrough
- `.gitattributes` — `export-ignore` if it stays unsupported
- `docs/` — a section stating GPU/CPU only, no NPU, with the measured
  overhead and the `.wslconfig` caveat if 1.3 hit it

## Closing out

Record the verdict in `TODONT.md` **even if the answer is no** — a documented
"we tested this and here is why not" is worth more to the next person than
silence. Then reply to #31 with the numbers.

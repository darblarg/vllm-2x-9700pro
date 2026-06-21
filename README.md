# Getting vLLM Working on Dual AMD Radeon AI PRO R9700 GPUs

**Versions this guide was tested against:**

| Component | Version |
|---|---|
| GPU / architecture | AMD Radeon AI PRO R9700 — gfx1201 (RDNA4) |
| vLLM | 0.23.0+rocm723 |
| ROCm SDK (core + libraries) | 7.13.0 |
| RCCL (AMD's multi-GPU communication library) | 2.28.3 |
| PyTorch | 2.10.0 (installed automatically with vLLM) |
| flash-attn | 2.8.3 |
| Base OS (Docker image) | Ubuntu 24.04 |

This stack changes a lot, some of these patches may not be required on newer builds.

## Who this is for

You have two AMD Radeon AI PRO R9700 cards (32GB of memory each, 64GB total) and you want them to work together to serve one large AI model through vLLM, a popular tool for running language models quickly. Running both cards together on one model is called **tensor parallelism**, or **TP** for short. People usually write this as `TP=2` when two GPUs are involved.

This guide is long because the short version doesn't work. The R9700 uses a very new GPU design (codename **gfx1201**, part of the RDNA4 family), and AMD's software stack for running AI models, called **ROCm**, has not fully caught up to this hardware yet. Several things are broken or missing by default, and they need to be fixed before vLLM will run well — or at all — across two cards.

The good news: every problem below has a known cause and a known fix. None of it requires deep GPU programming knowledge. You just need to follow the steps in order and understand *why* each one matters, so that when something goes wrong, you can tell which step to go back to.

A fair warning up front: this stuff moves fast. ROCm, vLLM, and the AMD-specific libraries they depend on are all under active development. Some exact version numbers in this guide will be out of date by the time you read it. The *reasoning* will still be useful even after the *numbers* change — that's why this guide explains the "why," not just the "do this."

## What you'll end up with

A Docker container running vLLM, splitting one model across both GPUs, answering requests over a normal web API (the same API format OpenAI uses, so most chat tools can connect to it directly). With a 27-billion-parameter model in FP8 format (a compact number format explained later), expect somewhere around 50–65 tokens per second once everything is set up and tuned correctly. "Tokens per second" is roughly how fast the model writes — a token is usually a word or part of a word.

---

## Table of contents

1. Check your hardware before you do anything else
2. Two things people get wrong with multi-GPU Docker setups
3. Why a plain, off-the-shelf vLLM image won't work
4. Building a patched vLLM image
5. Setting up the Docker Compose service
6. Launching it and knowing what "working" looks like
7. The big one: making two GPUs talk to each other without crashing
8. Picking a model and a number format
9. Squeezing out more speed
10. Performance

---

## 1. Check your hardware before you do anything else

Before installing anything, confirm Linux and the AMD driver can actually see both cards correctly. A surprising number of "this software is broken" problems turn out to be a driver issue or a card that isn't seated all the way.

```bash
rocm-smi                     # Lists each GPU — confirm you see two, each near 32GB
rocminfo | grep -A5 gfx      # Both cards should report "gfx1201"
ls /dev/dri/                 # Should show renderD128 and renderD129
```

If you only see one GPU, or the architecture name isn't `gfx1201`, stop here. Fix the driver or hardware issue first. Nothing in this guide will help if the operating system can't see both cards correctly in the first place.

One more thing worth checking if you just added a second GPU to a system that used to have one: adding a card can shift how Linux numbers your PCI devices, and that occasionally renames your network interface (for example, `eth0` becomes `eth1`, or `enp5s0` becomes something else). Run `ip link show` and make sure your network configuration still matches what's actually there. This has nothing to do with vLLM, but it's a classic "wait, why did my network just break" moment after a hardware change, and it's worth ruling out early.

## 2. Two things people get wrong with multi-GPU Docker setups

**Render nodes vs. card nodes.** Linux exposes each GPU to Docker containers in two different ways. The `renderD*` devices (like `renderD128`) are for compute work — the kind vLLM does. The `card*` devices (like `card0`) control actual video output to a monitor. When you tell Docker which GPU devices to pass into a container, **always use the renderD nodes, never the card nodes.** Passing a card node into a container can crash your desktop session, especially under Wayland. The symptom looks like "my whole desktop just died," which makes it a confusing thing to connect back to a Docker setting.

**Shared memory for GPU-to-GPU communication.** When two GPUs need to coordinate (which is the entire point of tensor parallelism), the software that handles that coordination needs a chunk of shared memory to pass messages through. Docker's default shared memory size is 64 megabytes, which is nowhere near enough. If you don't increase it, you'll get strange, hard-to-diagnose failures the moment both GPUs try to work together — not a clear "out of memory" message, just a crash. Set this explicitly:

```yaml
shm_size: '4gb'
```

Four gigabytes is comfortably more than needed, but it costs you nothing to over-allocate it, so don't try to find the exact minimum.

## 3. Why a plain, off-the-shelf vLLM image won't work

This is the part that catches people off guard, because it really seems like it should just work. vLLM officially supports AMD GPUs through ROCm. AMD publishes ROCm Docker images. So why doesn't installing vLLM and pointing it at your GPUs just work?

Because the R9700 is new enough that several pieces of the software stack haven't been updated to recognize it yet. None of these show up as a clear error message — they just quietly make things slower or, in one case, crash in a loop. Here's what's actually going on, in plain terms:

**Problem 1: The fast attention path doesn't know your GPU exists.**
vLLM has a library called **AITER** that provides faster versions of the core math operations a language model needs (the most important one is called "attention," which is most of what a model spends its time on while generating text). AITER has a fast mode called "unified attention." The code that decides whether to turn this fast mode on checks the GPU's chip family — and as of this writing, it doesn't check for the R9700's chip family (`gfx1201`) at all. So it silently falls back to a slower attention path.

**Problem 2: Some models hang for twenty minutes on startup.**
Certain newer model designs use a technique called Gated DeltaNet (you'll see "GDN" in logs). One of the math kernels these models need was written with a very specific memory layout that matches Nvidia's high-end "Hopper" GPUs. On the R9700, that exact layout doesn't compile correctly. It doesn't crash outright — it just hangs during a one-time startup tuning step, for anywhere up to twenty-some minutes, while spamming a confusing, unrelated-looking warning about "shared memory broadcast." If you've ever started a vLLM container, waited fifteen minutes, assumed it was broken, and killed it — this is very likely why. 

**Problem 3: The "pick the next word" step can crash outright.**
AITER also includes a faster version of the sampling step — the part that actually picks which word comes next. That faster version only compiles for AMD's data-center chips (the "CDNA" family, used in cards like the MI300). On the R9700, trying to use it causes a hard crash. If your container is set to restart automatically on failure, this turns into an infinite crash loop that.

Each problem requires small, specific edits to vLLM's own Python source code after it's installed. This guide applies patches to address each one when building the Docker image.

## 4. Building a patched vLLM image

Here is a complete Dockerfile that installs vLLM from AMD's official ROCm build index and applies the necessary fixes. The version numbers below were confirmed working at the time this guide was written — check whether newer ones are available, but see the warning in section 7 before you assume "newer is always better" on this hardware.

```dockerfile
# Build context should be the folder containing this Dockerfile and the
# patch script below (apply_gfx12x_patch.py).

ARG GFX_TARGET=gfx120X-all
ARG GFX_ARCH=gfx1201
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/rocm
ARG VLLM_VERSION=0.23.0+rocm723
ARG FLASH_ATTN_VERSION=2.8.3
ARG ROCM_SDK_CORE_VERSION=7.13.0
ARG ROCM_SDK_LIBRARIES_VERSION=7.13.0

# ---------------------------------------------------------------------------
# Stage 1: install ROCm and vLLM
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS base

ARG GFX_TARGET
ARG GFX_ARCH
ARG VLLM_WHEEL_URL
ARG VLLM_VERSION
ARG FLASH_ATTN_VERSION
ARG ROCM_SDK_CORE_VERSION
ARG ROCM_SDK_LIBRARIES_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/vllm
ENV PATH=/opt/vllm/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    libatomic1 libdrm2 libdrm-amdgpu1 libelf1 libgfortran5 libgomp1 \
    libjpeg-turbo8 libnuma1 libopenmpi3t64 \
    git curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/vllm \
    && python -m pip install --upgrade pip setuptools wheel uv

# Pulls vLLM, PyTorch, and friends from AMD/vLLM's prebuilt ROCm wheel index,
# rather than building from source. "fastapi<0.137" works around an unrelated
# FastAPI bug that breaks every HTTP request — see section 10.
RUN uv pip install \
    --find-links ${VLLM_WHEEL_URL}/vllm/ \
    --find-links ${VLLM_WHEEL_URL}/torch/ \
    --find-links ${VLLM_WHEEL_URL}/torchvision/ \
    --find-links ${VLLM_WHEEL_URL}/torchaudio/ \
    --find-links ${VLLM_WHEEL_URL}/triton/ \
    --find-links ${VLLM_WHEEL_URL}/triton-kernels/ \
    --find-links ${VLLM_WHEEL_URL}/amdsmi/ \
    --find-links ${VLLM_WHEEL_URL}/amd-aiter/ \
    --find-links ${VLLM_WHEEL_URL}/flash-attn/ \
    vllm==${VLLM_VERSION} flash-attn==${FLASH_ATTN_VERSION} "fastapi[standard]<0.137" \
    && uv pip install --no-deps torch-c-dlpack-ext

# Installs the actual ROCm GPU libraries, targeted specifically at the R9700's
# chip family.
RUN uv pip install --no-deps \
    --index-url https://repo.amd.com/rocm/whl/${GFX_TARGET}/ \
    rocm-sdk-core==${ROCM_SDK_CORE_VERSION} \
    rocm-sdk-libraries-gfx120X-all==${ROCM_SDK_LIBRARIES_VERSION} \
    rocm-sdk-devel==${ROCM_SDK_CORE_VERSION} \
    && python - <<'PY'
import os, site, tarfile
site_packages = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
tar_path = os.path.join(site_packages, "rocm_sdk_devel", "_devel.tar")
with tarfile.open(tar_path) as archive:
    archive.extractall(os.path.abspath(site_packages))
os.remove(tar_path)
PY

ENV SP=/opt/vllm/lib/python3.12/site-packages
ENV ROCM_PATH=${SP}/_rocm_sdk_devel
ENV ROCM_HOME=${SP}/_rocm_sdk_devel
ENV HIP_PATH=${SP}/_rocm_sdk_devel
ENV HIP_HOME=${SP}/_rocm_sdk_devel
ENV HIP_DEVICE_LIB_PATH=${SP}/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
ENV DEVICE_LIB_PATH=${SP}/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
ENV CPATH=${SP}/_rocm_sdk_devel/include:${CPATH}
ENV LIBRARY_PATH=${SP}/_rocm_sdk_devel/lib:${LIBRARY_PATH}
ENV PYTHONPATH=${SP}/_rocm_sdk_core/share/amd_smi
ENV LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/llvm/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_libraries_gfx120X_all/lib:${LD_LIBRARY_PATH}

ENV HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_ROCM_GCN_ARCH=${GFX_ARCH} \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} \
    AMDGPU_TARGETS=${GFX_ARCH} \
    GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 \
    TOKENIZERS_PARALLELISM=false

# Sanity check: fail the build immediately if the install is broken, instead
# of finding out three steps later.
RUN python -c 'import torch, vllm, vllm._C, vllm._rocm_C, flash_attn; print("vLLM", vllm.__version__, "/ ROCm", torch.version.hip, "- native extensions OK")'

ENTRYPOINT ["vllm", "serve"]

# ---------------------------------------------------------------------------
# Stage 2: apply the gfx1201/R9700 fixes from section 3
# ---------------------------------------------------------------------------
FROM base AS patched

COPY apply_gfx12x_patch.py /tmp/apply_gfx12x_patch.py
RUN python /tmp/apply_gfx12x_patch.py
```

Build it with:

```bash
docker build -t vllm-rocm-r9700:latest .
```


## The patch script itself

Save this next to the Dockerfile as `apply_gfx12x_patch.py`. It makes a small number of targeted text replacements inside vLLM's own installed files to address the three problems. :

```python
"""
Applies gfx1201/R9700 fixes to an installed vLLM. Uses plain text
replacement rather than a patch file, so it still works across small
version differences in vLLM's source.
"""
import os, site, py_compile

sp = next(p for p in site.getsitepackages() if p.endswith("site-packages"))

def patch(rel_path, old, new, description):
    full = os.path.join(sp, rel_path)
    content = open(full).read()
    if old not in content:
        raise AssertionError(f"Expected pattern not found in {rel_path}")
    open(full, "w").write(content.replace(old, new, 1))
    print(f"  patched: {rel_path} — {description}")

print("Applying gfx1201/R9700 patch...")

# Fix 1 of 3 (problem 1): tell vLLM's AITER eligibility check that gfx1201
# exists, so it stops assuming only AMD's data-center chips can use AITER.
patch(
    "vllm/_aiter_ops.py",
    "        from vllm.platforms.rocm import on_mi3xx\n\n        return on_mi3xx()\n",
    "        from vllm.platforms.rocm import on_gfx12x, on_mi3xx\n\n        return on_mi3xx() or on_gfx12x()\n",
    "recognize gfx1201 as AITER-eligible",
)

# Fix 1, continued: even after "eligible," vLLM picks an attention backend
# by priority order, and the older default backend still wins by default.
# This moves unified attention to the top of the list specifically on gfx1201.
patch(
    "vllm/platforms/rocm.py",
    "    if is_aiter_found_and_supported():\n"
    "        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n",
    "    if is_aiter_found_and_supported():\n"
    "        if on_gfx12x():\n"
    "            backends.insert(0, AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n"
    "        else:\n"
    "            backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)\n",
    "give unified attention top priority on gfx1201",
)

# Fix 1, continued: same idea, applied to the image-understanding (vision)
# attention path for multimodal models.
patch(
    "vllm/platforms/rocm.py",
    "        if rocm_aiter_ops.is_enabled() and on_gfx9():\n",
    "        if rocm_aiter_ops.is_mha_enabled() and (on_gfx9() or on_gfx12x()):\n",
    "extend vision attention backend check to gfx1201",
)
patch(
    "vllm/v1/attention/backends/rocm_aiter_fa.py",
    "        from vllm.platforms.rocm import on_mi3xx\n\n        return on_mi3xx()\n",
    "        from vllm.platforms.rocm import on_gfx12x, on_mi3xx\n\n        return on_mi3xx() or on_gfx12x()\n",
    "extend compute-capability check to gfx1201",
)
patch(
    "vllm/v1/attention/backends/rocm_aiter_fa.py",
    "                _MIN_HEAD_SIZE_FOR_LL4MI = 64\n"
    "                use_unified_attention = self.head_size < _MIN_HEAD_SIZE_FOR_LL4MI\n",
    "                _MIN_HEAD_SIZE_FOR_LL4MI = 64\n"
    "                from vllm.platforms.rocm import on_gfx12x\n"
    "                use_unified_attention = (\n"
    "                    self.head_size < _MIN_HEAD_SIZE_FOR_LL4MI or on_gfx12x()\n"
    "                )\n",
    "force unified attention kernel on gfx1201 (the alternative doesn't work on RDNA4)",
)

# Fix 2 (problem 2): the GDN startup-hang kernel. The original code uses a
# memory layout tuned for Nvidia Hopper GPUs unconditionally. This adds a
# branch so non-Nvidia GPUs use the older, compatible layout instead.
patch(
    "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
    "from vllm.triton_utils import tl, triton\n\nfrom .index import prepare_chunk_indices",
    "from vllm.triton_utils import tl, triton\nfrom vllm.platforms import current_platform\n\nfrom .index import prepare_chunk_indices",
    "import current_platform",
)
patch(
    "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
    "    IS_VARLEN: tl.constexpr,\n    USE_G: tl.constexpr,\n):\n",
    "    IS_VARLEN: tl.constexpr,\n    USE_G: tl.constexpr,\n    CAST_K_TRANS: tl.constexpr,\n):\n",
    "add a flag to choose the memory layout",
)
patch(
    "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
    "        b_kb = b_k * b_beta[:, None]\n"
    "        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))\n",
    "        b_kb = b_k * b_beta[:, None]\n"
    "        if CAST_K_TRANS:\n"
    "            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))  # Nvidia Hopper layout\n"
    "        else:\n"
    "            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))  # AMD-compatible layout\n",
    "use the AMD-compatible layout unless running on Nvidia",
)
patch(
    "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
    "        K=K,\n        BT=BT,\n    )\n    return A\n",
    "        K=K,\n        BT=BT,\n        CAST_K_TRANS=current_platform.is_cuda(),\n    )\n    return A\n",
    "pick the layout automatically based on the GPU vendor",
)

# Fix 3 (problem 3): don't use AITER's "pick the next word" kernel on
# gfx1201 — it's only compiled for AMD's data-center chips and crashes here.
# Fall back to the plain, always-available implementation instead.
patch(
    "vllm/v1/sample/ops/topk_topp_sampler.py",
    "            try:\n"
    "                import aiter.ops.sampling  # noqa: F401\n\n"
    "                self.aiter_ops = torch.ops.aiter\n"
    '                logger.info_once(\n                    "Using aiter sampler on ROCm (lazy import, sampling-only)."\n                )\n'
    "                self.forward = self.forward_hip\n"
    "            except ImportError:\n",
    "            try:\n"
    "                import aiter.ops.sampling  # noqa: F401\n"
    "                from vllm.platforms.rocm import on_gfx12x\n\n"
    "                self.aiter_ops = torch.ops.aiter\n"
    "                if on_gfx12x():\n"
    "                    self.forward = self.forward_native\n"
    "                else:\n"
    '                    logger.info_once(\n                        "Using aiter sampler on ROCm (lazy import, sampling-only)."\n                    )\n'
    "                    self.forward = self.forward_hip\n"
    "            except ImportError:\n",
    "use the safe sampler implementation on gfx1201",
)

# Double-check every file we touched still parses as valid Python.
for rel in [
    "vllm/_aiter_ops.py",
    "vllm/platforms/rocm.py",
    "vllm/v1/attention/backends/rocm_aiter_fa.py",
    "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
    "vllm/v1/sample/ops/topk_topp_sampler.py",
]:
    py_compile.compile(os.path.join(sp, rel), doraise=True)
    print(f"  syntax OK: {rel}")

print("Patch applied successfully.")
```

If a `patch()` call fails with "Expected pattern not found," it almost always means the version of vLLM you installed has changed that piece of code since this guide was written. Open the file it's complaining about, look at the surrounding code, and adjust the old/new text to match — the *intent* of each fix (described in the comments above) will still apply even if the exact wording of the source code has shifted.

## 5. Setting up the Docker Compose service

Here's a complete service definition. Replace the placeholder paths and token with your own.

```yaml
services:
  vllm:
    container_name: vllm
    image: vllm-rocm-r9700:latest
    ports:
      - "8001:8000"
    environment:
      # Only "unified attention" is turned on. The other AITER features
      # (MOE, MLA, FP8 matrix multiply, etc.) either aren't validated on
      # this chip yet or are known not to work — leave them off.
      - VLLM_ROCM_USE_AITER=1
      - VLLM_ROCM_USE_AITER_MHA=0
      - VLLM_ROCM_USE_AITER_MLA=0
      - VLLM_ROCM_USE_AITER_MOE=0
      - VLLM_ROCM_USE_AITER_LINEAR=0
      - VLLM_ROCM_USE_AITER_FP8BMM=0
      - VLLM_ROCM_USE_AITER_FP4BMM=0
      - VLLM_ROCM_USE_AITER_TRITON_GEMM=0
      - VLLM_ROCM_USE_AITER_RMSNORM=0
      - VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
      # Needed if you use speculative decoding (section 9) — without this,
      # it runs noticeably slower on RDNA4.
      - GPU_MAX_HW_QUEUES=1
      - HF_HOME=/models/hf-cache
      - HF_TOKEN=${HF_TOKEN}        # put your real token in a .env file, never in this file directly
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
      - /dev/dri/renderD129:/dev/dri/renderD129
      - /dev/kfd:/dev/kfd
    group_add:
      - "44"
      - "991"
    shm_size: '4gb'
    volumes:
      - ./models:/models
      - ./compile-cache:/root/.cache/vllm
      - ./triton-cache:/root/.triton/cache
    command:
      - Qwen/Qwen3.6-27B-FP8           # swap for whatever model you want to run
      - --tensor-parallel-size
      - "2"
      - --max-model-len
      - "131072"
      - --kv-cache-dtype
      - fp8_e4m3
      - --gpu-memory-utilization
      - "0.88"
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --max-num-seqs
      - "32"
    restart: unless-stopped
```

A few of these settings deserve explanation:

- **`./compile-cache` and `./triton-cache` volumes** — vLLM compiles GPU code the first time it sees a model, and that compilation is slow. These two folders save the compiled results to disk so you don't pay that cost again on every restart. Without them, every restart is much slower than it needs to be.
- **`--gpu-memory-utilization 0.88`** — this tells vLLM to use up to 88% of each GPU's memory for the model and its cache. You may not need the same amount of overhead for other apps.
- **`--kv-cache-dtype fp8_e4m3`** — the "KV cache" is the model's short-term memory of the conversation so far. Storing it in a compact 8-bit format instead of the usual 16-bit format roughly doubles how much conversation history fits in memory, at a small, usually unnoticeable quality cost.

## 6. Launching it and knowing what "working" looks like

```bash
docker compose up -d
docker compose logs -f vllm
```

Expect the whole startup to take a few minutes — even with everything cached, figure five to ten minutes is normal. The first time you start a model (before the compile cache has anything in it), it can take twenty minutes or more. It's a one-time cost for performance mapping.

Watch the logs for this line, which confirms the attention fix from section 3 actually took effect:

```
Using aiter unified attention for RocmAiterUnifiedAttentionImpl
```

If you instead see a line mentioning a different backend (`MHA`, `flash_attn`, or similar) where you expected unified attention, the patch didn't apply, or didn't apply to the file vLLM actually loaded. Double check you built the patched image (not an old cached one) and that the container is actually running it.

If startup seems to hang for a long time around model loading, with a repeating warning about a "shared memory broadcast block" not being found — that's very likely the GDN startup-hang issue from section 3, problem 2, and it should eventually finish on its own. Give it the full twenty-plus minutes before assuming it's actually stuck. If your image has the patch applied correctly, this step usually finishes in well under a minute instead.

## 7. The big one: making two GPUs talk to each other without crashing

This is the step that causes the most frustration, and it deserves its own section.

### A quick vocabulary note

When two GPUs split one model between them, they constantly need to share partial results back and forth. The library that handles this is, historically, called **NCCL** (Nvidia's name for it). AMD's version is called **RCCL**, and it's built to be a drop-in replacement. Here's the confusing part: PyTorch's code still refers to the communication backend as `"nccl"` even when it's actually using RCCL underneath on an AMD system. If you see "nccl" in an error message on an all-AMD machine, that's normal — it doesn't mean anything is misconfigured. It's just old naming that stuck around.

### The error you'll probably see

If TP=2 isn't working, you'll typically get something like:

```
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

This message is famously unhelpful — "unhandled error" doesn't tell you what actually went wrong. As of this writing, there have been at least two distinct, unrelated bugs on R9700-class hardware that both produce this exact message:

**Bug A — a missing tuning entry (mostly fixed, has a known workaround).**
RCCL picks an internal communication strategy based on a lookup table that maps each GPU chip type to a tuning profile. For a while, that table simply had no entry for gfx1201, so RCCL silently used a tuning profile meant for an entirely different, much older AMD data-center chip. The mismatch caused multi-GPU communication to freeze completely — both GPUs would sit at 100% usage, doing nothing, forever.

The workaround, confirmed by multiple people independently and by AMD's own vLLM-ROCm team:

```bash
NCCL_PROTO=Simple vllm serve <model> --tensor-parallel-size 2
```

This tells RCCL to use a simpler, older communication method that doesn't depend on the missing tuning entry. It has no real downside for typical inference use.

**Bug B — a newer, different crash (still being tracked upstream as of this writing).**
On some newer development builds of ROCm and RCCL, the symptom changes from a silent freeze to an outright crash, with an error like `hipErrorIllegalState`, happening the instant the GPUs try to run their very first joint operation. Critically, **`NCCL_PROTO=Simple` does not fix this one** — it's a different bug further down in the same library, in the code path that actually launches GPU work, not the code path that picks a tuning profile. If you've already tried `NCCL_PROTO=Simple` and you're still crashing immediately, you are very likely looking at this second bug, not the first one.

### How to tell which bug you have

Strip vLLM out of the picture entirely and test GPU-to-GPU communication directly. Save this as `tp_test.py`:

```python
import torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

x = torch.ones(8, device=f"cuda:{rank}")
dist.all_reduce(x)
torch.cuda.synchronize()

print(f"rank {rank} OK -> {x.tolist()}")
dist.destroy_process_group()
```

Run it with:

```bash
torchrun --nproc_per_node=2 tp_test.py
```

- If this **hangs forever** with both GPUs pinned at high usage — that's Bug A. Try `NCCL_PROTO=Simple`.
- If this **crashes immediately** with a clear fatal error — that's Bug B, or something else entirely lower-level. `NCCL_PROTO=Simple` won't help here.

This script is also a great way to confirm your fix actually worked, separately from vLLM, before you spend time debugging vLLM-specific settings that were never the problem.

### If you're hitting Bug B

This is the harder one, because as of this writing it doesn't have a single agreed-upon fix — it's an active upstream issue. A few things worth trying, roughly in order of how easy they are to test:

1. **Use prebuilt, official wheels instead of building ROCm from a nightly source branch.** If you're using a toolbox or install script that compiles ROCm and RCCL from a rolling development branch, you're at the mercy of whatever commit happened to be on that branch the day you built it — and these branches do regress. The Dockerfile in section 4 installs from AMD's published wheel index at pinned version numbers instead, which is a more stable target. Two systems can report the exact same RCCL *version number* and still behave differently if they were built from different commits on the same nightly branch — version numbers on fast-moving nightly software are not as meaningful as they look.
2. **Try a newer kernel and a newer base OS, if you're not already on one.** RDNA4 firmware and driver support is landing across kernel releases on a rolling basis. Hardware this new sometimes genuinely needs a newer kernel than what your distribution shipped with a year ago.
3. **Test the `amdgpu.noretry` kernel boot parameter both ways.** This is a low-confidence suggestion — it's a real, low-level setting that affects how the GPU driver handles certain memory faults, and it's occasionally relevant to obscure multi-GPU compute crashes, but it isn't a guaranteed fix for this specific bug. It's cheap to test: check your current setting in `/proc/cmdline`, and if you can, try it set the other way.
4. **Check whether the bug has a fix yet.** This is genuinely a moving target. Search for the exact error text from your logs before spending hours debugging something that may already be fixed in a newer package.

## 8. Picking a model and a number format

Models come in different numeric formats (sometimes called "quantizations") that trade off memory use, speed, and quality. The two you'll see most often for a card like this are:

- **FP8** — an 8-bit floating-point format. This has a working, reasonably fast kernel path on the R9700 already, even though it hasn't been specifically hand-tuned for this card yet (more on that in the next section).
- **AWQ** — a different 4-bit integer format, popular because it produces smaller files. I didn't find an AWQ version that works on this card. I saw 5 tokens per second. If you see numbers that bad, suspect the quantization format before you suspect your hardware.

There are other formats out there, these are the results from the 2 I tried. 

## 9. Squeezing out more speed

**Turn on speculative decoding if your model supports it.** Some models ship with a small built-in "draft" component that guesses a few words ahead, which the main model then checks all at once instead of generating one word at a time. When the guesses are right (which is often), this is essentially free speed. Look for "MTP" in a model's documentation, and add this to your launch command if it's supported:

```yaml
- --speculative-config
- '{"method":"mtp","num_speculative_tokens":3}'
```

Combined with `GPU_MAX_HW_QUEUES=1` from section 5 (which this feature specifically needs on this hardware), this can take a model from around 35 tokens per second to 55–65 tokens per second.


## 10. Performance

These numbers are for a 27-billion-parameter model in FP8 format, with TP=2 across both cards, once everything above is working correctly. Use them to sanity-check your own results — if you're far below these, something above is still misconfigured.

| Setup | Tokens per second |
|---|---|
| TP=2, FP8, unified attention working, no speculative decoding | ~35–45 |
| TP=2, FP8, unified attention working, with speculative decoding (MTP) | ~55–65 |
| Single GPU, same model, smaller quantized version | ~15–20 |
| TP=2, AWQ format, kernel not properly supported (the failure mode from section 8) | ~5 |

That last row is included on purpose. If your numbers look like that, you have not discovered that your hardware is broken — you've discovered the AWQ kernel-support gap. Switch formats before you start questioning your motherboard.

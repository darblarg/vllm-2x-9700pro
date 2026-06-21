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

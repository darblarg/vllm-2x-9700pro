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

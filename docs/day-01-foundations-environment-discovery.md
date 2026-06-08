# Foundations & environment discovery

_Day **1** of 7 · DGX OS 7 · GB10 Blackwell_

Before deploying anything, you map the terrain: what DGX OS 7 gives you, how to read the GPU, and how to confirm that the Docker → NVIDIA runtime → GB10 chain works end to end.

<!-- Optional image slot:
![Day 1: Foundations and environment discovery](assets/images/day-01-foundations.png)
-->

## Overview & objectives

> [!NOTE]
> **Learning objectives**
>
> - Explain what DGX OS 7 provides and why all AI work on it is container-based.
> - Read every line of `nvidia-smi` and know your total VRAM.
> - Confirm Docker and the NVIDIA Container Toolkit are active.
> - Run the single test that proves a container can reach the GB10 GPU.
> - Log in to the NGC catalog and understand the GB10 / sm_121 hardware reality.

> [!NOTE]
> **Prerequisites**
>
> Physical access to the Lenovo ThinkStation PGX (Type 30KL) with DGX OS 7 installed, a terminal, and network access. No prior AI experience is assumed.

> [!NOTE]
> **Files involved**
>
> None yet — Day 1 is pure exploration of the pre-installed environment.

## What DGX OS 7 actually is

**Concept**

DGX OS 7 is NVIDIA's tuned Linux for DGX/Spark-class machines. It ships everything you need so you never assemble a GPU stack by hand:

| Component | What it gives you |
| --- | --- |
| Ubuntu 24.04 LTS base | A standard Linux you already know — all the usual tools are present. |
| CUDA pre-installed | NVIDIA's GPU toolkit, so code can run on the card. On this machine CUDA 13.0 is the system level. |
| Recent NVIDIA drivers | The layer between the OS and the GB10 Blackwell GPU — already tuned and current. |
| Docker + NVIDIA Container Toolkit | Lets containers reach the GPU. This is the key to the whole ecosystem. |

> [!CAUTION]
> **Warning — No host installs**
>
> **The golden rule.** On DGX/PGX you do *not* install AI software directly on the host — no `pip install` or `apt install` of frameworks. Everything runs in Docker containers, so the host stays clean and NVIDIA-managed.

**AI software** (what you want) → **Docker image** (packaged) → **Container** (running) → **GPU access** (--gpus all)

**Hands-on**

```bash
# See the OS version
$ cat /etc/os-release

# Confirm the ARM64 architecture — decides which Docker images you can use
$ uname -m
```

> [!TIP]
> **Expected output**
>
> ```text
> aarch64
> ```

> [!NOTE]
> **Explanation**
>
> `uname -m` returns `aarch64` (ARM64), **not** `x86_64`. Remember this: many third-party container images are x86-only and will fail with `exec format error` on this machine. You will prefer official NVIDIA NGC images and ARM-aware community images throughout.

## Reading nvidia-smi — your GPU dashboard

**Hands-on**

```bash
# Instant overview
$ nvidia-smi
```

> [!TIP]
> **Expected output**
>
> ```text
> +-------------------------------------------------------------------------+
> | NVIDIA-SMI 570.xx     Driver: 570.xx        CUDA Version: 13.0          |
> +------------------+-----------------------+------------------------------+
> | GPU  Name        | Bus-Id        Disp.A  | Volatile Uncorr. ECC         |
> |  0   GB10 Blackwl| 00000000:01:00.0 Off  |                    0          |
> +------------------+-----------------------+------------------------------+
> | Fan  Temp  Perf  | Pwr:Usage/Cap         | Memory-Usage                 |
> | N/A  42C   P0    | 38W / 300W            | 1024MiB / 98304MiB           |
> +------------------+-----------------------+------------------------------+
> ```

**Explanation**

| Field | What it tells you |
| --- | --- |
| Driver / CUDA Version | The driver and CUDA level. You pick NGC containers to match these. |
| GB10 Blackwell | The GPU name — confirms the driver recognises the card. |
| Temp (42C) | Idle temperature. Normal idle 40–60°C; up to ~85°C under load is fine. |
| Pwr 38W / 300W | Power draw / cap. Idle ~40W; ~150–250W during inference. |
| 1024MiB / 98304MiB | VRAM used / total. The **total** is what decides which models you can load. |

> [!IMPORTANT]
> **Deep dive — Unified memory on GB10**
>
> On the GB10 the GPU and CPU share one large pool of **unified memory**. PyTorch reports about 130.6 GB of GPU-visible memory on this system. That is why you can load very large models that would never fit on a consumer card — there is no separate small VRAM bank. Keep an eye on the total figure; it is your real budget across every service you run.

> [!TIP]
> **Hands-on**
>
> ```bash
> # Live monitoring — keep this open in a second terminal all day
> $ watch -n 2 nvidia-smi
>
> # Just memory, machine-readable
> $ nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
>
> # Continuous per-column monitor (memory + utilisation)
> $ nvidia-smi dmon -s mu
> ```

## Docker & the NVIDIA Container Toolkit

Docker is pre-installed on DGX OS 7. You only verify it is active and that the NVIDIA runtime is registered.

**Hands-on**

```bash
$ docker --version
$ docker compose version
$ systemctl status docker
```

> [!TIP]
> **Expected output**
>
> ```text
> ● docker.service - Docker Application Container Engine
>      Loaded: loaded (/lib/systemd/system/docker.service)
>      Active: active (running) since ...
> ```

```bash
# If it is inactive, start and enable it
$ sudo systemctl start docker && sudo systemctl enable docker
```

**Hands-on**

```bash
# Is the NVIDIA runtime registered with Docker?
$ docker info | grep -i runtime
```

> [!TIP]
> **Expected output**
>
> ```text
> Runtimes: io.containerd.runc.v2 nvidia runc
>                                  ^^^^^^ "nvidia" must be present
> ```

```bash
# If "nvidia" is missing (rare on DGX OS), reconfigure it
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker
```

> [!NOTE]
> **Explanation**
>
> The NVIDIA Container Toolkit is the bridge that lets a container see the GPU. Without the `nvidia` runtime registered, the `--gpus all` flag (and the Compose `deploy` block you meet on Day 2) does nothing.

## The fundamental test: GPU inside a container

> [!NOTE]
> **Concept**
>
> This is *the* test that validates the whole chain. Success means exactly one thing: the `nvidia-smi` table prints from **inside** a container.

**Hands-on**

```bash
# Run nvidia-smi from INSIDE a fresh container
$ docker run --rm --gpus all \
    nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 \
    nvidia-smi
```

> [!TIP]
> **Expected output**
>
> ```text
> +-------------------------------------------------------------------------+
> | NVIDIA-SMI 570.xx   <- this table printing from the container = success |
> +-------------------------------------------------------------------------+
> |  0   GB10 Blackwell ...                                                  |
> +-------------------------------------------------------------------------+
> ```

> [!NOTE]
> **Explanation**
>
> If you see that table, the full chain works: Docker → NVIDIA Container Toolkit → GB10. The options mean:
>
> - `--rm` — delete the container automatically when it exits (no zombie containers).
> - `--gpus all` — expose every GPU to the container; without it the container cannot see the GB10.
> - `nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04` — NVIDIA's minimal CUDA base image. NGC publishes CUDA 13 ARM64 images; Docker Hub does **not** (it only has x86 for CUDA 13).

> [!TIP]
> **Hands-on**
>
> ```bash
> # Variant: drop into an interactive shell inside a CUDA container
> $ docker run --rm -it --gpus all \
>     nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 bash
> root@container:/# nvidia-smi
> root@container:/# exit
> ```

## The NGC catalog — official NVIDIA containers

> [!NOTE]
> **Concept**
>
> NGC (NVIDIA GPU Cloud, `catalog.ngc.nvidia.com`) is the hub of containers tuned for your hardware. It is free with an account, and NGC images are the ones with proper GB10 / ARM64 support.

> [!TIP]
> **Hands-on**
>
> ```bash
> # 1. Create a free account at catalog.ngc.nvidia.com
> # 2. Settings -> API Keys -> Generate Personal Key
> # 3. Log in from the terminal:
> $ docker login nvcr.io
>   Username: $oauthtoken          <- literally this text, including the $
>   Password: <your-NGC-API-key>   <- pasted from the NGC web UI
> ```

> [!WARNING]
> **Common issue — $oauthtoken is literal**
>
> The username is always the literal string `$oauthtoken` (dollar sign included). It is an NGC quirk, not a shell variable — do not substitute anything for it.

**Hands-on**

```bash
# Pull the reference PyTorch container (Blackwell + ARM64)
$ docker pull nvcr.io/nvidia/pytorch:25.10-py3

# Confirm PyTorch sees the GB10
$ docker run --rm --gpus all \
    nvcr.io/nvidia/pytorch:25.10-py3 \
    python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

> [!TIP]
> **Expected output**
>
> ```text
> =============
> == PyTorch ==
> =============
>
> NVIDIA Release 25.10
> PyTorch Version 2.9.0a0+145a3a7
> ...
>
> NOTE: The SHMEM allocation limit is set to the default of 64MB. This may be
> insufficient for PyTorch. NVIDIA recommends the use of the following flags:
> docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...
>
> CUDA available: True
> GPU: NVIDIA GB10
> VRAM: 130.6 GB
> ```
>
> The NVIDIA banner and SHMEM note are normal for this quick container test; the important lines are CUDA available, GPU name, and VRAM.

> [!NOTE]
> **Explanation**
>
> Seeing `True` and the GB10 name from a PyTorch NGC container is the final confirmation that everything is wired correctly. `nvcr.io/nvidia/pytorch:25.10-py3` is the container you reuse to build ComfyUI (Day 6) and AudioCraft (Day 7).

## Deep dive — the GB10 / sm_121 reality

> [!IMPORTANT]
> **Deep dive**
>
> The GB10 is very recent hardware and that has practical consequences you must understand before choosing any container:
>
> - The GB10 uses the Blackwell architecture with compute capability **sm_121** — a value unique to the GB10 (desktop RTX 50-series are sm_100/sm_120).
> - Full sm_121 support in NGC containers only lands from **NGC 25.10+** (October 2025). Earlier containers may start but JIT kernels for sm_121 fail or fall back to CPU.
> - From NGC 25.10, `sm_120 PTX` is bundled and JIT-compiles to sm_121 on first launch — so PyTorch runs natively on the GB10 with no manual work.

**Explanation**

| Container | Status on GB10 | Notes |
| --- | --- | --- |
| `nvcr.io/nvidia/pytorch:25.10-py3` | ✅ confirmed | Reference image. PyTorch 2.9.0a0, CUDA 13.0, sm_120+PTX. |
| `nvcr.io/nvidia/pytorch:25.12-py3` | ✅ confirmed | PyTorch 2.10.0a0. Note: `torchaudio` is missing. |
| `nvcr.io/nvidia/cuda:13.0.1-*-ubuntu24.04` | ✅ confirmed | Only NGC ships CUDA 13 ARM64; Docker Hub is x86-only. |
| `nvcr.io/nvidia/vllm:25.10-py3` | ✅ confirmed | Official vLLM for DGX Spark / GB10 (Day 5). |
| `ollama/ollama:latest` | ✅ confirmed | Native ARM64; the engine you start on Day 3. |
| `nvcr.io/nvidia/pytorch:≤24.10-py3` | ⚠ partial | No sm_121; set `TORCH_CUDA_ARCH_LIST="12.0+PTX"` as a workaround. |
| Third-party ComfyUI images (yanwk, ai-dock) | ❌ x86 only | Build your own from NGC (Day 6). |

> [!WARNING]
> **Common issue — Known GB10 gaps (handled later)**
>
> Two known gaps you will meet later — both have clean workarounds in this course:
>
> - `torchaudio` is absent from NGC 25.10 on aarch64. Harmless for image generation (a one-line stub), and for audio it is replaced by a full on-disk stub built in the AudioCraft Dockerfile (Day 7).
> - Flash Attention 3 supports sm_90–sm_110 only; on GB10 use Flash Attention 2 (binary-compatible via sm_120) or PyTorch's native `scaled_dot_product_attention`.

> [!TIP]
> **Hands-on**
>
> Optional — the NGC CLI lets you search and pull from the terminal (download the **ARM64 Linux** build):
>
> ```bash
> # Configure once with your NGC API key
> $ ngc config set
>
> # Browse what is available
> $ ngc registry image list 'nvidia/*'
> $ ngc registry image info nvidia/pytorch
> $ ngc registry image list --query "llm"
> ```

## Checkpoint & exercises

> [!TIP]
> **Checkpoint**
>
> - [ ] `nvidia-smi` runs without error — `nvidia-smi`
> - [ ] I noted my total VRAM in MiB — `nvidia-smi --query-gpu=memory.total --format=csv,noheader`
> - [ ] Architecture is aarch64 — `uname -m  →  aarch64`
> - [ ] Docker is active (running) — `systemctl status docker`
> - [ ] "nvidia" appears in the Docker runtimes — `docker info | grep -i runtime`
> - [ ] nvidia-smi works from INSIDE a container — `docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi`
> - [ ] docker compose version prints — `docker compose version`
> - [ ] NGC account created and docker login nvcr.io succeeded — `docker login nvcr.io`
> - [ ] PyTorch NGC recognises the GB10 (True) — `torch.cuda.is_available() → True`

> [!TIP]
> **Exercise**
>
> 1. Open `watch -n 2 nvidia-smi` in a second terminal, then in the first run the PyTorch container test. Watch VRAM rise and fall.
> 2. Run `docker run --rm -it --gpus all nvcr.io/nvidia/pytorch:25.10-py3 bash` and, inside, print `torch.version.cuda` and `torch.__version__`.
> 3. Write down which two known GB10 gaps exist and which day each one is solved in.

> [!NOTE]
> **Recap**
>
> You have a healthy environment, you can read `nvidia-smi`, Docker reaches the GB10, and you understand why NGC 25.10+ matters for sm_121. **Day 2** turns this into a real stack: the complete `docker-compose.yml` and how containers get the GPU through Compose.

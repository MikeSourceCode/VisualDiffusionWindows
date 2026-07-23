"""Shared generation engine.

Consolidates the backend/VRAM detection, architecture handling, pipeline
loading, LoRA application and the ``generate`` call that were previously
duplicated across every standalone script. Feature tabs call into this module
instead of holding implicit module-level state.
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance
from compel import CompelForSDXL
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput

import safety

from .config import AppConfig, LoraSpec


warnings.filterwarnings("ignore", message=r".*Token indices sequence length is longer than the specified maximum sequence length.*")
warnings.filterwarnings("ignore", message=r".*There are modules in .* that should be kept in float32.*")
warnings.filterwarnings("ignore", message=r".*Casting directly with `to\(\) can lead to inconsistent results.*")


# --- Safety (operator-configurable via AppConfig) ---
# Fraction of denoise steps at which the single early NSFW check fires. The
# default is 0.5 (mid-denoise); the operator can override this in config.


class Backend(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


class VRAMState(Enum):
    HIGH_VRAM = 4
    NORMAL_VRAM = 3
    SHARED = 5
    DISABLED = 0


def detect_backend() -> Backend:
    if torch.cuda.is_available():
        return Backend.ROCM if hasattr(torch.version, "hip") and torch.version.hip else Backend.CUDA
    if torch.backends.mps.is_available():
        return Backend.MPS
    return Backend.CPU


def get_device_string(backend: Backend) -> str:
    if backend in (Backend.CUDA, Backend.ROCM):
        return "cuda"
    if backend == Backend.MPS:
        return "mps"
    return "cpu"


def should_use_fp16(backend: Backend) -> bool:
    return backend != Backend.CPU


def get_vae_dtype(backend: Backend) -> torch.dtype:
    """Dtype for a BUILT-IN VAE on this backend.

    Matches the known-good standalone scripts (e.g. image_to_image_diffusion.py):
    on MPS the built-in SDXL VAE MUST decode in fp32, because in fp16 it overflows
    and produces a uniform gray (NaN) image. ROCm uses bf16 when available, else
    fp32; CUDA uses fp16 (where the built-in VAE is stable); CPU uses fp32.
    """
    if backend == Backend.MPS:
        return torch.float32
    if backend == Backend.ROCM:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if backend == Backend.CPU:
        return torch.float32
    return torch.float16


def custom_vae_dtype(backend: Backend) -> torch.dtype:
    """Dtype for a CUSTOM (user-supplied) VAE. Force fp32 everywhere: custom VAE
    weights overflow in fp16 on MPS and yield NaN/gray output.
    """
    return torch.float32


def get_vram_state(backend: Backend, total_mb: float) -> VRAMState:
    if backend == Backend.CPU:
        return VRAMState.DISABLED
    if backend == Backend.MPS:
        return VRAMState.SHARED
    if total_mb < 8192:
        raise RuntimeError(f"VRAM too low: {total_mb:.0f}MB")
    return VRAMState.HIGH_VRAM if total_mb >= 16384 else VRAMState.NORMAL_VRAM


def clear_cache(backend: Backend):
    if backend == Backend.MPS:
        torch.mps.empty_cache()
    elif backend == Backend.CUDA:
        torch.cuda.empty_cache()


def detect_vram_mb(backend: Backend) -> float:
    if backend in (Backend.CUDA, Backend.ROCM) and torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0] / (1024 * 1024)
    return 0.0


def _hook_scheduler(pipe):
    """Steal the predicted x0 each step for live previews (matches scripts).

    The clean latent is stored on BOTH the pipe and the scheduler. Storing it on
    the scheduler means a txt2img sibling that shares this scheduler instance can
    still read the value via ``pipe_instance.scheduler`` even though the closure
    below targets the original pipe.
    """
    original_step = pipe.scheduler.step

    def hooked_step(*args, **kwargs):
        kwargs["return_dict"] = True
        step_output = original_step(*args, **kwargs)
        clean = getattr(step_output, "pred_original_sample", None)
        pipe.current_clean_latent = clean
        try:
            pipe.scheduler.current_clean_latent = clean
        except Exception:
            pass
        return EulerDiscreteSchedulerOutput(prev_sample=step_output.prev_sample)

    pipe.scheduler.step = hooked_step


def place_unet(pipe, backend: Backend, vram_state: VRAMState):
    device = torch.device(get_device_string(backend))
    if vram_state != VRAMState.DISABLED:
        pipe.unet.to(device)
    return device


def place_controlnet(pipe, backend: Backend, vram_state: VRAMState):
    device = torch.device(get_device_string(backend))
    if vram_state != VRAMState.DISABLED and hasattr(pipe, "controlnet"):
        pipe.controlnet.to(device)
    return device


def place_vae(pipe, backend: Backend, vram_state: VRAMState):
    device = torch.device(get_device_string(backend))
    if vram_state != VRAMState.DISABLED:
        pipe.vae.to(device)
    return device


def place_clip(pipe, backend: Backend, vram_state: VRAMState):
    device = torch.device(get_device_string(backend))
    if vram_state != VRAMState.DISABLED:
        pipe.text_encoder.to(device)
        if hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2.to(device)
    return device


def _configure_vae(pipe, backend: Backend, custom_vae: bool = False):
    # A custom (user-supplied) VAE stays fp32 to avoid the MPS fp16 overflow that
    # yields NaN/gray output. The default built-in SDXL VAE also needs fp32 on MPS
    # (it collapses to gray in fp16); get_vae_dtype encodes that per-backend rule,
    # mirroring the known-good standalone scripts. Slice + tile so a full-res
    # decode fits in shared memory and stays stable.
    #
    # IMPORTANT: on MPS, calling .to(dtype=...) ALONE moves the VAE to CPU and
    # drops the device, which then causes "input(cpu)/weight(mps) device mismatch"
    # errors and gray output. We must pass BOTH device and dtype so the VAE stays
    # on the compute device in the correct dtype.
    vae_dev = torch.device(get_device_string(backend))
    pipe.vae.to(device=vae_dev, dtype=get_vae_dtype(backend))
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


def _lock_vae_fp32(pipe, backend: Backend) -> None:
    """Prevent the pipeline from silently recasting the VAE to fp16 on MPS.

    After load we force the VAE to fp32 (see _configure_vae), but diffusers'
    Img2Img/ControlNet pipelines call ``self.vae.to(self.dtype)`` during the run
    (``self.dtype`` is the fp16 ``torch_dtype`` passed to ``from_single_file``),
    which on MPS collapses the decode to uniform gray NaN. Patching ``vae.to`` so
    it always keeps the compute device + fp32 dtype keeps BOTH the init-image
    encode and the final decode in fp32, matching the known-good txt2img path.
    """
    if backend != Backend.MPS:
        return
    vae = pipe.vae
    vae_dev = torch.device(get_device_string(backend))
    orig_to = vae.to

    def _keep_fp32(device=None, dtype=None, *args, **kwargs):
        return orig_to(device=vae_dev, dtype=torch.float32)

    vae.to = _keep_fp32


def default_vae_path(vae_dir: str) -> Optional[str]:
    """Pick the VAE to use when the user selected "built-in".

    Returning None keeps the checkpoint's BAKED-IN SDXL VAE, which is what the
    known-good standalone scripts (e.g. image_to_image_diffusion.py) used and
    which produces the correct, non-rainbowy colors. The earlier switch to the
    bundled sdxl-vae-fp16-fix VAE introduced a color shift ("rainbowy" output)
    and is no longer used as the default. A custom VAE is still honored when the
    user picks one explicitly in the sidebar.
    """
    return None


def load_base_pipeline(model_path: str, vae_path: Optional[str], backend: Backend,
                       architecture: str, img2img: bool = False):
    clear_cache(backend)
    if architecture == "SDXL":
        pipe_class = StableDiffusionXLImg2ImgPipeline if img2img else StableDiffusionXLPipeline
    else:
        pipe_class = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
    compute_dtype = torch.float16 if should_use_fp16(backend) else torch.float32

    custom_vae = bool(vae_path and os.path.exists(vae_path))
    if os.path.isdir(model_path):
        try:
            pipe = pipe_class.from_pretrained(model_path, torch_dtype=compute_dtype, use_safetensors=True)
        except ValueError:
            pipe = None
        if pipe is None:
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=compute_dtype, use_safetensors=True)
        if custom_vae:
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=custom_vae_dtype(backend))
            pipe.vae = vae
    else:
        if custom_vae:
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=custom_vae_dtype(backend))
            pipe = pipe_class.from_single_file(model_path, vae=vae, torch_dtype=compute_dtype,
                                               use_safetensors=True, upcast_vae=False)
        else:
            pipe = pipe_class.from_single_file(model_path, torch_dtype=compute_dtype,
                                               use_safetensors=True, upcast_vae=False)

    if architecture == "SD 1.5" and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

    _configure_vae(pipe, backend, custom_vae=custom_vae)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    _hook_scheduler(pipe)
    pipe._architecture = architecture
    return pipe


def load_controlnet_pipeline(model_path: str, vae_path: Optional[str], controlnet_path: str,
                             backend: Backend, architecture: str):
    from diffusers import ControlNetModel

    clear_cache(backend)
    compute_dtype = torch.float16 if should_use_fp16(backend) else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=compute_dtype)

    pipe_class = StableDiffusionXLControlNetImg2ImgPipeline if architecture == "SDXL" \
        else StableDiffusionControlNetImg2ImgPipeline

    custom_vae = bool(vae_path and os.path.exists(vae_path))
    if custom_vae:
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=custom_vae_dtype(backend))
        pipe = pipe_class.from_single_file(model_path, controlnet=controlnet, vae=vae,
                                           torch_dtype=compute_dtype, use_safetensors=True,
                                           upcast_vae=False)
    else:
        pipe = pipe_class.from_single_file(model_path, controlnet=controlnet,
                                           torch_dtype=compute_dtype, use_safetensors=True,
                                           upcast_vae=False)

    _configure_vae(pipe, backend, custom_vae=custom_vae)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    _hook_scheduler(pipe)
    pipe._architecture = architecture
    return pipe


def load_loras_into_unet(pipe, lora_specs: List[LoraSpec], lora_dir: str, backend: Backend):
    # Keep only named LoRAs with positive weight whose file actually exists.
    lora_specs = [
        (name, weight) for (name, weight) in lora_specs
        if name and weight > 0.0 and os.path.exists(os.path.join(lora_dir, name))
    ]
    if not lora_specs:
        pipe.unload_lora_weights()
        return
    pipe.unload_lora_weights()
    clear_cache(backend)
    adapter_names = []
    current_arch = getattr(pipe, "_architecture", "SDXL")
    for i, (name, weight) in enumerate(lora_specs):
        state_dict, network_alphas, metadata = pipe.lora_state_dict(
            lora_dir, weight_name=name, unet_config=pipe.unet.config, return_lora_metadata=True
        )
        lora_arch = _detect_lora_architecture(state_dict)
        if lora_arch and lora_arch != current_arch:
            print(f"[LoRA] Skipping {name}: architecture mismatch "
                  f"(LoRA is {lora_arch}, pipeline is {current_arch})")
            continue
        if weight != 1.0:
            state_dict = {k: (v * weight if k.endswith("lora_up.weight") else v)
                          for k, v in state_dict.items()}
        adapter_name = f"lora{i + 1}"
        try:
            pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet,
                                     adapter_name=adapter_name, metadata=metadata, _pipeline=pipe)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"[LoRA] Skipping {name}: weight shapes do not match "
                      f"the current {current_arch} UNet ({e})")
                continue
            raise
        adapter_names.append(adapter_name)
    if adapter_names:
        pipe.unet.set_adapter(adapter_names)


def _detect_lora_architecture(state_dict: dict) -> Optional[str]:
    """Heuristic architecture detection from LoRA weight tensor shapes.

        Returns ``"SDXL"`` if weights look like 1x1 conv LoRA adapters
        (4D tensors), ``"SD15"`` if they look like linear LoRA adapters
        (2D tensors), or ``None`` if it cannot be determined.
    """
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 4:
            return "SDXL"
        if v.ndim == 2:
            return "SD15"
    return None


def _build_conditioning(pipe, prompt: str, negative_prompt: str, architecture: str) -> dict:
    if architecture == "SDXL":
        try:
            compel = CompelForSDXL(pipe=pipe)
            cond = compel(prompt, negative_prompt=negative_prompt)
            return {
                "prompt_embeds": cond.embeds,
                "pooled_prompt_embeds": cond.pooled_embeds,
                "negative_prompt_embeds": cond.negative_embeds,
                "negative_pooled_prompt_embeds": cond.negative_pooled_embeds,
            }
        except Exception as e:
            print(f"[Compel] failed ({e}), using raw prompt string")
    return {"prompt": prompt, "negative_prompt": negative_prompt}


def _decode_latents(pipe, latents, backend: Backend) -> Image.Image:
    """Decode SD/SDXL latents to a PIL image.

    Mirrors the standalone scripts: divide by the VAE scaling factor, decode in
    the VAE's OWN dtype (built-in VAE is fp32 on MPS to avoid the fp16 NaN/gray
    collapse; a custom VAE also stays fp32), sanitize NaN/Inf, then map
    [-1,1] -> [0,1]. No contrast re-normalization is applied; previews come from
    the scheduler's predicted clean latent (``current_clean_latent``), which
    already renders as a clean image rather than raw diffusion noise.
    """
    vae_dev = pipe.vae.device
    # Decode in the dtype dictated by the backend rule (fp32 on MPS) rather than
    # pipe.vae.dtype: even with _lock_vae_fp32 the safest contract is to decode
    # in the known-good dtype, never fp16 on MPS (which yields gray NaN).
    vae_dtype = get_vae_dtype(backend)
    architecture = getattr(pipe, "_architecture", "SDXL")
    vae_scale = 0.13025 if architecture == "SDXL" else 0.18215
    with torch.no_grad():
        vae_input = (latents / vae_scale).to(dtype=vae_dtype, device=vae_dev)
        vae_out = pipe.vae.decode(vae_input).sample
        vae_out = torch.nan_to_num(vae_out, nan=0.0, posinf=1.0, neginf=-1.0)
        img = (vae_out / 2 + 0.5).clamp(0, 1)
        np_img = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img)


def _as_txt2img(pipe, architecture: str):
    """Return a txt2img pipeline sharing ``pipe``'s already-loaded components.

    The base pipeline is loaded as an Img2Img class so uploads work, but that
    class cannot run without an ``image``. For pure txt2img we reuse the same
    weights via a sibling pipeline, cached on the source pipe so we build it at
    most once.
    """
    cached = getattr(pipe, "_txt2img_sibling", None)
    if cached is not None:
        return cached
    if architecture == "SDXL":
        if isinstance(pipe, (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline)):
            return pipe
        if isinstance(pipe, StableDiffusionXLControlNetImg2ImgPipeline):
            sibling = StableDiffusionXLControlNetPipeline(**pipe.components)
        else:
            sibling = StableDiffusionXLPipeline(**pipe.components)
    else:
        if isinstance(pipe, (StableDiffusionPipeline, StableDiffusionControlNetPipeline)):
            return pipe
        if isinstance(pipe, StableDiffusionControlNetImg2ImgPipeline):
            sibling = StableDiffusionControlNetPipeline(**pipe.components)
        else:
            sibling = StableDiffusionPipeline(**pipe.components)
    sibling._architecture = architecture
    # The sibling shares the base pipe's scheduler instance, which is already
    # hooked; the clean latent is read via the scheduler in the callback.
    try:
        pipe._txt2img_sibling = sibling
    except Exception:
        pass
    return sibling


def generate(pipe, backend: Backend, vram_state: VRAMState, config: AppConfig,
              prompt: str, negative_prompt: str,
              init_image: Optional[Image.Image] = None,
              control_image: Optional[Image.Image] = None,
              control_scale: Optional[float] = None,
              preview_callback: Optional[Callable[[Image.Image, int], None]] = None,
              early_safety_check: Optional[bool] = None,
              lora_dir: str = "") -> Image.Image:
    """Run a generation using an ``AppConfig`` as the single source of truth.

    Works for txt2img and img2img (``init_image``), and optionally ControlNet
    (``control_image`` + ``control_scale``).
    """
    architecture = getattr(pipe, "_architecture", config.architecture)
    do_img2img = init_image is not None and config.strength > 0.0

    # A base pipeline is loaded as an Img2Img class (so uploads work), but that
    # class REQUIRES an ``image`` argument. For pure txt2img (no init_image) we
    # must run through a txt2img pipeline instead. Build a sibling that shares
    # the already-loaded components (no extra VRAM, no reload) and cache it on
    # the pipe so repeat generations are free.
    if not do_img2img:
        pipe = _as_txt2img(pipe, architecture)

    # Move weights onto the compute device (matches the standalone scripts).
    place_unet(pipe, backend, vram_state)
    place_vae(pipe, backend, vram_state)
    place_clip(pipe, backend, vram_state)
    place_controlnet(pipe, backend, vram_state)
    # Ensure the pipeline itself tracks the correct device so internal
    # preprocessing (e.g. control_image on MPS) moves tensors consistently.
    device = torch.device(get_device_string(backend))
    if vram_state != VRAMState.DISABLED:
        pipe.to(device)
    # Lock the VAE to fp32 on MPS so the pipeline's internal .to(self.dtype)
    # recast can't collapse img2img/ControlNet decode to gray NaN.
    _lock_vae_fp32(pipe, backend)

    # Pre-generation prompt gate (safety.py Layer 1 + 2).
    safety.check_prompt(prompt, negative_prompt)

    pipe_kwargs = _build_conditioning(pipe, prompt, negative_prompt, architecture)
    # Generator on CPU: seeding on MPS with the Karras Euler scheduler is what
    # previously hung generation; CPU seeding matches the known-good script path.
    # Resolve the seed once so callers can record the exact value used.
    used_seed = config.effective_seed()
    generator = torch.Generator(device="cpu").manual_seed(used_seed)

    strength = config.strength
    if do_img2img:
        print(f"[IMG2IMG] strength={strength} steps={config.steps}")
        pipe_kwargs.update({
            "image": init_image,
            "strength": strength,
            "num_inference_steps": config.steps,
            "guidance_scale": config.cfg_scale,
        })
    else:
        print(f"[TXT2IMG] {config.width}x{config.height}")
        pipe_kwargs.update({
            "num_inference_steps": config.steps,
            "guidance_scale": config.cfg_scale,
            "width": config.width,
            "height": config.height,
        })

    if control_image is not None:
        pipe_kwargs["control_image"] = control_image
        pipe_kwargs["controlnet_conditioning_scale"] = control_scale if control_scale is not None else config.controlnet_strength

    preview_every = config.preview_every(config.steps)
    # Early safety gate: a SINGLE CLIP NSFW check at the denoise fraction below
    # (step 4 of 8, step 8 of 16 at 0.5) instead of every step (slow) or only at
    # the end (wasted compute). The final output guard (safety.censor) still
    # always runs regardless. This fraction is operator-configurable.
    do_early_check = config.early_safety_check if early_safety_check is None else early_safety_check
    if config.early_safety_step > 0:
        early_step = config.early_safety_step
    else:
        early_step = max(1, round(config.steps * config.early_safety_step_frac))
    early_checked = {"done": False}

    def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
        if not preview_callback:
            return callback_kwargs
        # Prefer the scheduler's predicted clean latent (pred_original_sample),
        # which decodes to a clean image; fall back to the raw step latent.
        # NOTE: use an explicit `is None` check — `tensor or x` raises
        # "Boolean value of Tensor is ambiguous".
        latents = getattr(pipe_instance, "current_clean_latent", None)
        if latents is None:
            latents = getattr(getattr(pipe_instance, "scheduler", None), "current_clean_latent", None)
        if latents is None:
            latents = callback_kwargs.get("latents")
        try:
            preview = _decode_latents(pipe_instance, latents, backend)
            preview_callback(preview, step_index)
            # One-shot early NSFW check at the configured fraction.
            if do_early_check and not early_checked["done"] and step_index >= early_step:
                early_checked["done"] = True
                _, flagged = safety.censor_image(preview)
                if flagged:
                    raise SystemExit(
                        f"[SAFETY] Unsafe content detected at early step {step_index}; "
                        f"aborting before final output."
                    )
        except SystemExit:
            raise
        except Exception as e:
            print(f"[Preview] Step {step_index} failed: {e}")
        return callback_kwargs

    t0 = time.time()
    final_latents = pipe(
        **pipe_kwargs,
        generator=generator,
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent",
    ).images

    result = _decode_latents(pipe, final_latents, backend)
    if config.sharpness > 0:
        result = ImageEnhance.Sharpness(result).enhance(1.0 + config.sharpness)

    # Output guard (safety.py Layer 3).
    result = safety.censor(result)
    print(f"Generation finished in {time.time() - t0:.2f}s")
    # Record the exact seed used so callers can save reproducible filenames.
    try:
        result.info["seed"] = str(used_seed)
    except Exception:
        pass
    return result


def load_init_image_from_path(image_path: str, target_size=(832, 1472)) -> Image.Image:
    """Load and center-pad an image to ``target_size`` (default 9:16)."""
    if image_path.startswith("~"):
        image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Init image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    if img.size != target_size:
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))
        img = new_img
    return img

"""
Shared utilities for SAR-DDPM ultrasound despeckling scripts.
Leverages original SAR_DDPM codebase to eliminate redundancy.
"""

import sys, os, argparse, copy, time
import numpy as np
import torch
import cv2
from scipy.io import loadmat
from scipy.signal import hilbert
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Add SAR_DDPM to path for imports
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAR_DIR = os.path.join(PROJECT_DIR, "SAR_DDPM")
SCRIPTS_DIR = os.path.join(SAR_DIR, "scripts")
sys.path.insert(0, SAR_DIR)
sys.path.insert(0, SCRIPTS_DIR)

# Import from original SAR_DDPM codebase
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
)
from guided_diffusion.resample import UniformSampler


# ---------------------------------------------------------------------------
# Model Management (using original SAR_DDPM defaults)
# ---------------------------------------------------------------------------

def build_model_and_diffusion(device):
    """Create model+diffusion with exact SAR-DDPM configuration."""
    model_defaults = sr_model_and_diffusion_defaults()
    # Surcharges exactes du papier (cf. sarddpm_test.py ligne par ligne)
    model_defaults.update({
        "large_size":          256,
        "small_size":          64,
        "num_channels":        192,
        "num_res_blocks":      2,
        "num_heads":           4,
        "attention_resolutions": "32,16,8",
        "diffusion_steps":     1000,
        "noise_schedule":      "linear",
        "learn_sigma":         True,
        "resblock_updown":     True,
        "use_scale_shift_norm": True,
        "class_cond":          True,
        "use_fp16":            False,
        # DDIM respacing comme dans sarddpm_test.py
        "use_ddim":            False,
        "timestep_respacing":  "ddim25",
    })
    model, diffusion = sr_create_model_and_diffusion(
        **{k: model_defaults[k]
           for k in sr_model_and_diffusion_defaults().keys()}
    )
    return model, diffusion


def load_weights(model, path, device, strict=False):
    """Load weights with fallback for mismatched keys."""
    state = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(state, strict=strict)
        print(f"  Loaded {len(state)} params (strict={strict})")
    except RuntimeError:
        md = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in md and v.shape == md[k].shape}
        md.update(matched)
        model.load_state_dict(md)
        print(f"  Loaded {len(matched)}/{len(state)} params (partial — "
              f"{len(state)-len(matched)} unused keys in file, "
              f"{len(md)-len(matched)} model keys randomly initialised)")


# ---------------------------------------------------------------------------
# Image Processing Utilities (from original MATLAB equivalents)
# ---------------------------------------------------------------------------

def rf2bmode(rf, increase=0):
    """Python equivalent of src/rf2bmode.m."""
    rf = rf.astype(np.float64)
    if rf.ndim == 2:
        rf = rf[:, :, np.newaxis]
    out = np.zeros_like(rf)
    for i in range(rf.shape[2]):
        a = hilbert(rf[:, :, i], axis=0)
        b = 20 * np.log(np.abs(a) + increase + 1e-12)
        b -= b.min()
        mx = b.max()
        if mx > 0:
            b = 255.0 * b / mx
        out[:, :, i] = b
    return out.squeeze()


def calc_cnr(R1, R2):
    """Python equivalent of src/calc_CNR.m."""
    R1, R2 = R1.astype(np.float64), R2.astype(np.float64)
    m1, m2 = np.mean(R1), np.mean(R2)
    v1, v2 = np.std(R1)**2, np.std(R2)**2
    return 20 * np.log10(abs(m1-m2) / np.sqrt(v1+v2+1e-12) + 1e-12)


def load_gray(path):
    """Load any supported file as normalized grayscale uint8."""
    if path.endswith(".mat"):
        mat = loadmat(path)
        # Filter to numeric 2D+ arrays only
        keys = [k for k in mat.keys() if not k.startswith("__")]
        arr = None
        for k in keys:
            candidate = mat[k]
            if hasattr(candidate, 'dtype') and candidate.dtype.kind in ('f', 'u', 'i', 'c') \
               and candidate.ndim >= 2:
                arr = candidate
                break
        if arr is None:
            raise ValueError(f"No numeric 2D array found in {path}")
        if arr.dtype in [np.complex64, np.complex128]:
            arr = np.abs(arr)
        arr = arr.astype(np.float64)
        if arr.min() < 0 or (arr.max() - arr.min()) > 1000:
            arr = rf2bmode(arr)
        arr -= arr.min()
        if arr.max() > 0:
            arr = 255.0 * arr / arr.max()
        return arr.astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img


def to_tensor_3ch(gray_uint8, device):
    """Convert grayscale uint8 to 3-channel tensor normalized to [-1, 1]."""
    g256 = cv2.resize(gray_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)
    arr = np.repeat(g256[:, :, np.newaxis], 3, axis=2)
    arr = arr.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def tensor_to_gray(t):
    """Convert model output tensor to grayscale uint8."""
    s = t.cpu().float()[0].mean(dim=0)
    return ((s + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()


def tensor_to_gray_uint8(t):
    """Convert model output tensor (1,3,H,W) to grayscale uint8, as in sarddpm_test.py."""
    out = ((t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    out = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]  # HWC
    out = out[:, :, ::-1]  # RGB->BGR
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def save_img(path, arr):
    """Save image array, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr)


# ---------------------------------------------------------------------------
# Speckle Synthesis (using original SAR_DDPM implementation)
# ---------------------------------------------------------------------------

# Same fixed seed as in SAR_DDPM/guided_diffusion/image_datasets.py
_SPECKLE_RNG = np.random.RandomState(112311)


def add_speckle(clean_uint8, rng=None):
    """
    Multiplicative speckle synthesis — identical to image_datasets.py in SAR_DDPM.
    y = sqrt(intensity * Gamma(1,1)),  intensity = ((pixel+1)/256)^2
    """
    if rng is None:
        rng = _SPECKLE_RNG
    img = np.float32(clean_uint8)
    im1 = ((img + 1.0) / 256.0) ** 2
    gn = rng.gamma(size=im1.shape, shape=1.0, scale=1.0).astype(im1.dtype)
    noisy = np.sqrt(im1 * gn) * 256 - 1
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference Methods (using original SAR_DDPM cycle spinning)
# ---------------------------------------------------------------------------

def cycle_spinning_inference(model, diffusion, SR_tensor, device):
    """
    Exact cycle-spinning loop from SAR_DDPM/scripts/sarddpm_test.py.
    SR_tensor: (1, 3, 256, 256) in [-1, 1] — the speckled input.
    Returns denoised (1, 3, 256, 256) tensor.
    """
    single_img = SR_tensor.clone()
    _, _, max_r, max_c = single_img.size()
    N = 9
    sample_new = None
    count = 0

    for row in range(0, max_r, 100):
        for col in range(0, max_c, 100):
            val_inputv = single_img.clone()
            # Circular shift
            val_inputv[:, :, :row, :col]  = single_img[:, :, max_r-row:, max_c-col:]
            val_inputv[:, :, row:,  col:] = single_img[:, :, :max_r-row, :max_c-col]
            val_inputv[:, :, row:,  :col] = single_img[:, :, :max_r-row, max_c-col:]
            val_inputv[:, :, :row,  col:] = single_img[:, :, max_r-row:, :max_c-col]

            model_kwargs = {"SR": val_inputv.to(device), "HR": val_inputv.to(device)}
            sample = diffusion.p_sample_loop(
                model, (1, 3, 256, 256),
                clip_denoised=True, model_kwargs=model_kwargs, device=device,
            )

            if count == 0:
                sample_new = (1.0 / N) * sample
            else:
                sample_new[:, :, max_r-row:, max_c-col:] += (1.0/N) * sample[:, :, :row,  :col]
                sample_new[:, :, :max_r-row, :max_c-col] += (1.0/N) * sample[:, :, row:,  col:]
                sample_new[:, :, :max_r-row, max_c-col:] += (1.0/N) * sample[:, :, row:,  :col]
                sample_new[:, :, max_r-row:, :max_c-col] += (1.0/N) * sample[:, :, :row,  col:]
            count += 1

    return sample_new


def infer_no_cs(model, diffusion, SR_t, device):
    """Simple inference without cycle spinning (faster baseline)."""
    model_kwargs = {"SR": SR_t, "HR": SR_t}
    with torch.no_grad():
        return diffusion.p_sample_loop(
            model, (1, 3, 256, 256),
            clip_denoised=True, model_kwargs=model_kwargs, device=device, progress=True,
        )


def infer(model, diffusion, SR_t, device, use_cs=False):
    """Unified inference function supporting both modes."""
    with torch.no_grad():
        if use_cs:
            return cycle_spinning_inference(model, diffusion, SR_t, device)
        else:
            return infer_no_cs(model, diffusion, SR_t, device)


# ---------------------------------------------------------------------------
# Data Discovery and Processing
# ---------------------------------------------------------------------------

def discover_simu_data(data_dir):
    """Discover all simulated data pairs for processing."""
    simu_dir = os.path.join(data_dir, "simu")
    pairs = []
    
    for folder in sorted(os.listdir(simu_dir)):
        fp = os.path.join(simu_dir, folder)
        if not os.path.isdir(fp):
            continue
        files = set(os.listdir(fp))
        
        # Folders 1-4
        if "bmode_GT.png" in files and "bmode.png" in files:
            pairs.append(("bmode", os.path.join(fp, "bmode.png"), 
                         os.path.join(fp, "bmode_GT.png"), f"simu{folder}_bmode"))
        if "GT_rf.png" in files and "rf.png" in files:
            pairs.append(("rf", os.path.join(fp, "rf.png"),
                         os.path.join(fp, "GT_rf.png"), f"simu{folder}_rf"))
        # Folders 5-6
        if "US_GT.png" in files and "US_observed.png" in files:
            pairs.append(("US", os.path.join(fp, "US_observed.png"),
                         os.path.join(fp, "US_GT.png"), f"simu{folder}_US"))
        # Folder 7: no GT
        if "rf_humankidney.mat" in files:
            pairs.append(("kidney", os.path.join(fp, "rf_humankidney.mat"), 
                         None, f"simu{folder}_kidney"))
    
    return pairs


def discover_vivo_data(data_dir):
    """Discover all in-vivo data for processing."""
    vivo_dir = os.path.join(data_dir, "vivo")
    data = []
    
    for folder in sorted(os.listdir(vivo_dir)):
        fp = os.path.join(vivo_dir, folder)
        if not os.path.isdir(fp):
            continue
        files = set(os.listdir(fp))
        
        to_process = []
        if "bmode.png" in files:
            to_process.append(("bmode_png", os.path.join(fp, "bmode.png")))
        if "bmode.mat" in files:
            to_process.append(("bmode_mat", os.path.join(fp, "bmode.mat")))
        if "rf.mat" in files:
            to_process.append(("rf_mat", os.path.join(fp, "rf.mat")))
        if "data.mat" in files:
            to_process.append(("data_mat", os.path.join(fp, "data.mat")))
        
        for name, path in to_process:
            data.append((name, path, f"vivo/{folder}/{name}"))
    
    return data


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_image_metrics(gt, denoised):
    """Compute PSNR, SSIM, MSE between ground truth and denoised images."""
    m_psnr = psnr_fn(gt, denoised, data_range=255)
    m_ssim = ssim_fn(gt, denoised, data_range=255)
    m_mse = float(np.mean((gt.astype(float) - denoised.astype(float))**2))
    return {"PSNR": m_psnr, "SSIM": m_ssim, "MSE": m_mse}


def compute_cnr_metrics(input_img, denoised_img, q=64):
    """Compute CNR improvement metrics."""
    cnr_in = calc_cnr(input_img[q:2*q, q:2*q], input_img[2*q:3*q, 2*q:3*q])
    cnr_dn = calc_cnr(denoised_img[q:2*q, q:2*q], denoised_img[2*q:3*q, 2*q:3*q])
    return {
        "CNR_input": cnr_in, 
        "CNR_denoised": cnr_dn,
        "CNR_improvement": cnr_dn - cnr_in
    }


# ---------------------------------------------------------------------------
# Training Utilities (for finetuning)
# ---------------------------------------------------------------------------

def collect_gt_images(data_dir):
    """Gather all available ground-truth images from src/simu/."""
    imgs = []
    simu_dir = os.path.join(data_dir, "simu")
    
    for folder in sorted(os.listdir(simu_dir)):
        fp = os.path.join(simu_dir, folder)
        files = set(os.listdir(fp)) if os.path.isdir(fp) else set()
        for fname in ["bmode_GT.png", "GT_rf.png", "US_GT.png"]:
            if fname in files:
                img = cv2.imread(os.path.join(fp, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(img)
                    print(f"  GT: simu/{folder}/{fname}  {img.shape}")
    
    print(f"  Total GT images for training: {len(imgs)}")
    return imgs


def finetune_model(model, diffusion, device, gt_images,
                   epochs=100, lr=2e-5):
    model = model.to(device)
    sampler = UniformSampler(diffusion)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    rng = np.random.RandomState(42)
    if len(gt_images) > 2:
        val_imgs   = gt_images[-1:] 
        train_imgs = gt_images[:-1]
    else:
        val_imgs   = gt_images
        train_imgs = gt_images

    gts_train = [cv2.resize(g, (256, 256)) for g in train_imgs]
    gts_val   = [cv2.resize(g, (256, 256)) for g in val_imgs]

    iters_per_epoch = max(50, 200 // max(len(gts_train), 1))
    best_val_loss = float("inf")
    best_state    = None
    patience      = 15   # early stopping
    no_improve    = 0

    print(f"\nFinetuning : {epochs} epochs × {iters_per_epoch} iters")
    print(f"  Train images : {len(gts_train)} | Val images : {len(gts_val)}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for _ in range(iters_per_epoch):
            idx    = rng.randint(len(gts_train))
            clean  = gts_train[idx]
            noisy  = add_speckle(clean, rng=rng)

            clean_t = to_tensor_3ch(clean, device)
            noisy_t = to_tensor_3ch(noisy, device)

            t, weights = sampler.sample(1, device)
            losses = diffusion.training_losses(
                model, clean_t, t,
                model_kwargs={"SR": noisy_t, "HR": clean_t}
            )
            loss = (losses["loss"] * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean in gts_val:
                noisy   = add_speckle(clean, rng=rng)
                clean_t = to_tensor_3ch(clean, device)
                noisy_t = to_tensor_3ch(noisy, device)
                t, weights = sampler.sample(1, device)
                losses = diffusion.training_losses(
                    model, clean_t, t,
                    model_kwargs={"SR": noisy_t, "HR": clean_t}
                )
                val_loss += (losses["loss"] * weights).mean().item()
        val_loss /= max(len(gts_val), 1)

        avg_train = train_loss / iters_per_epoch
        print(f"  Epoch {epoch+1:4d}/{epochs}  "
              f"train={avg_train:.6f}  val={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping à l'epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def create_report_header(title, weights_path, extra_info=""):
    """Create standardized report header."""
    lines = [
        "=" * 70,
        title,
        f"Weights: {weights_path}",
    ]
    if extra_info:
        lines.append(extra_info)
    lines.extend(["=" * 70, ""])
    return lines


def add_metrics_to_report(report_lines, label, metrics):
    """Add metrics to report in standardized format."""
    if "PSNR" in metrics:  # Has GT
        report_lines.extend([
            f"{label}:",
            f"  PSNR:  {metrics['PSNR']:.4f} dB",
            f"  SSIM:  {metrics['SSIM']:.4f}",
            f"  MSE:   {metrics['MSE']:.4f}", "",
        ])
    elif "CNR_input" in metrics:  # CNR metrics
        report_lines.extend([
            f"{label}:",
            f"  CNR_input:        {metrics['CNR_input']:.4f} dB",
            f"  CNR_denoised:     {metrics['CNR_denoised']:.4f} dB",
            f"  CNR_improvement:  {metrics['CNR_improvement']:+.4f} dB", "",
        ])
    else:  # No metrics
        report_lines.extend([f"{label}: no GT available", ""])


def save_report(report_lines, output_path):
    """Save report to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved: {output_path}")

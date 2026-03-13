"""
Script 3 — Ultrasound Finetuning + Inference (Refactored)
==========================================================
Fine-tunes the SAR-DDPM model on ultrasound ground-truth images (from src/simu/)
using the same speckle synthesis from SAR_DDPM/guided_diffusion/image_datasets.py,
then runs inference on all simulated and in-vivo datasets.

Training uses the diffusion training_losses() from gaussian_diffusion.py and
the UniformSampler from resample.py — the same primitives as the original
SAR_DDPM training script.

Finetuned weights are saved to: weights/sar_ddpm_us_finetuned.pt
Results are saved to: results_us_finetuned/
"""

import sys, os, argparse
import numpy as np
import torch
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAR_DIR = os.path.join(PROJECT_DIR, "SAR_DDPM")

# Import shared utilities
from shared_utils import (
    build_model_and_diffusion, load_weights, infer,
    tensor_to_gray, save_img, load_gray, to_tensor_3ch,
    discover_simu_data, discover_vivo_data, compute_image_metrics,
    compute_cnr_metrics, create_report_header, add_metrics_to_report,
    save_report, collect_gt_images, finetune_model
)

WEIGHTS_IN = os.path.join(SAR_DIR, "weights", "64_256_upsampler.pt")
WEIGHTS_OUT = os.path.join(PROJECT_DIR, "weights", "sar_ddpm_us_finetuned.pt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results_us_finetuned")
DATA_DIR = os.path.join(PROJECT_DIR, "src")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cycle_spinning", action="store_true")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load existing finetuned weights")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, diffusion = build_model_and_diffusion(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights_dir = os.path.dirname(WEIGHTS_OUT)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)

    if args.skip_training and os.path.exists(WEIGHTS_OUT):
        print(f"\nLoading finetuned weights: {WEIGHTS_OUT}")
        load_weights(model, WEIGHTS_OUT, device)
    else:
        print(f"\nLoading base SAR weights: {WEIGHTS_IN}")
        load_weights(model, WEIGHTS_IN, device)

        if not args.skip_training:
            gt_images = collect_gt_images(DATA_DIR)
            model = finetune_model(model, diffusion, device, gt_images,
                                   epochs=args.epochs, lr=args.lr)
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"\nFinetuned weights saved: {WEIGHTS_OUT}")
        else:
            print("\nWARNING: no finetuned weights found, using base SAR weights.")

    model.to(device).eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = create_report_header(
        "SAR-DDPM on Ultrasound — WITH Finetuning",
        f"Base weights: {WEIGHTS_IN}",
        f"Finetuned weights: {WEIGHTS_OUT}\nEpochs: {args.epochs}  lr: {args.lr}"
    )
    report_lines.append("SIMULATED DATA (PSNR / SSIM / MSE):")

    print("\n" + "="*60 + "\nSIMULATED DATA\n" + "="*60)
    simu_pairs = discover_simu_data(DATA_DIR)
    
    for kind, noisy_path, clean_path, label in simu_pairs:
        print(f"\n  [{label}]")
        out_folder = os.path.join(OUTPUT_DIR, "simu", label.split("_")[0])
        os.makedirs(out_folder, exist_ok=True)

        noisy = load_gray(noisy_path)
        SR_t = to_tensor_3ch(noisy, device)
        denoised_t = infer(model, diffusion, SR_t, device, args.cycle_spinning)
        denoised = tensor_to_gray_uint8(denoised_t)
        n256 = noisy if noisy.shape[0] == 256 else cv2.resize(noisy, (256, 256))

        save_img(os.path.join(out_folder, f"{label.split('_')[1]}_noisy.png"), n256)
        save_img(os.path.join(out_folder, f"{label.split('_')[1]}_denoised.png"), denoised)

        if clean_path:
            gt = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            gt256 = cv2.resize(gt, (256, 256))
            save_img(os.path.join(out_folder, f"{label.split('_')[1]}_GT.png"), gt256)
            
            metrics = compute_image_metrics(gt256, denoised)
            print(f"    PSNR={metrics['PSNR']:.2f} dB  SSIM={metrics['SSIM']:.4f}  MSE={metrics['MSE']:.1f}")
            add_metrics_to_report(report_lines, label, metrics)
        else:
            add_metrics_to_report(report_lines, label, {})

    report_lines.extend(["", "IN VIVO DATA (CNR in dB):"])
    print("\n" + "="*60 + "\nIN VIVO DATA\n" + "="*60)
    vivo_data = discover_vivo_data(DATA_DIR)
    
    for name, path, label in vivo_data:
        print(f"\n  [{label}]")
        out_folder = os.path.join(OUTPUT_DIR, "vivo", label.split("/")[1])
        os.makedirs(out_folder, exist_ok=True)

        try:
            img = load_gray(path)
        except Exception as e:
            print(f"    Skipping {name}: {e}")
            report_lines.append(f"  {label}: skipped ({e})")
            continue
            
        SR_t = to_tensor_3ch(img, device)
        denoised_t = infer(model, diffusion, SR_t, device, args.cycle_spinning)
        denoised = tensor_to_gray_uint8(denoised_t)
        in256 = img if img.shape[0] == 256 else cv2.resize(img, (256, 256))

        save_img(os.path.join(out_folder, f"{name}_input.png"), in256)
        save_img(os.path.join(out_folder, f"{name}_denoised.png"), denoised)

        cnr_metrics = compute_cnr_metrics(in256, denoised)
        print(f"    CNR: {cnr_metrics['CNR_input']:.2f} -> {cnr_metrics['CNR_denoised']:.2f} ({cnr_metrics['CNR_improvement']:+.2f} dB)")
        report_lines.append(
            f"  {label}: CNR_in={cnr_metrics['CNR_input']:.4f}  CNR_out={cnr_metrics['CNR_denoised']:.4f}  "
            f"delta={cnr_metrics['CNR_improvement']:+.4f}"
        )

    save_report(report_lines, os.path.join(OUTPUT_DIR, "metrics_report.txt"))
    print(f"\nResults: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import numpy as np
import torch

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from models.coatnet import load_coatnet
from models.cnn import load_cnn_model
from face_extractor import (
    analyze_video_frames,
    preprocess_for_coatnet,
    preprocess_for_cnn
)

def run_prediction(video_path, model_type="coatnet", weights_path=None, max_frames=20, threshold=0.5):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    print("\n=======================================================")
    print("[*] DEEPFAKE VIDEO FORENSIC INFERENCE")
    print("=======================================================")
    print(f"Target Video     : {video_path}")
    print(f"Model Engine     : {model_type.upper()}")
    print(f"Weights Path     : {weights_path or 'Untrained baseline / heuristic mode'}")
    print(f"Frames to sample : {max_frames}")
    print(f"Decision cut-off : {threshold}")
    print(f"-------------------------------------------------------")

    print("[1/3] Extracting frames & detecting faces via Haar Cascade...")
    records = analyze_video_frames(video_path, max_frames=max_frames)
    if not records:
        print("❌ Error: No frames could be parsed from the video.")
        sys.exit(1)

    detected_faces = sum(1 for r in records if r["face_detected"])
    print(f"      Parsed {len(records)} frames. Faces detected: {detected_faces}/{len(records)} ({(detected_faces/len(records))*100:.1f}%)")

    print("[2/3] Loading neural network...")
    fake_probs = []

    if model_type.lower() == "coatnet":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, has_weights = load_coatnet(weights_path, device=device)
        print(f"      CoAtNet running on {device}. Pretrained weights loaded: {has_weights}")

        print("[3/3] Running frame-by-frame inference...")
        for r in records:
            tensor = preprocess_for_coatnet(r["face_crop_bgr"], img_size=224).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                p_fake = float(probs[1]) if len(probs) > 1 else float(probs[0])
            
            if not has_weights:
                lap_score = min(1.0, max(0.0, (150.0 - r["laplacian_var"]) / 150.0))
                p_fake = float(np.clip(0.4 * p_fake + 0.6 * lap_score, 0.05, 0.95))
            
            fake_probs.append(p_fake)

    else:
        model, has_weights = load_cnn_model(weights_path)
        print(f"      4-Stage Keras CNN loaded. Pretrained weights: {has_weights}")

        print("[3/3] Running frame-by-frame inference...")
        for r in records:
            batch = preprocess_for_cnn(r["face_crop_bgr"], img_size=128)
            pred = model.predict(batch, verbose=0)
            p_fake = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0][1])

            if not has_weights:
                lap_score = min(1.0, max(0.0, (150.0 - r["laplacian_var"]) / 150.0))
                p_fake = float(np.clip(0.4 * p_fake + 0.6 * lap_score, 0.05, 0.95))

            fake_probs.append(p_fake)

    avg_fake_prob = float(np.mean(fake_probs))
    is_fake = avg_fake_prob >= threshold

    print(f"\n=======================================================")
    print(f"[+] FORENSIC RESULTS SUMMARY")
    print(f"=======================================================")
    if is_fake:
        print(f"[!] FINAL VERDICT        : DEEPFAKE DETECTED (MANIPULATED)")
        print(f"    Manipulation Score   : {avg_fake_prob * 100:.2f}%")
    else:
        print(f"[+] FINAL VERDICT        : AUTHENTIC (REAL VIDEO)")
        print(f"    Authenticity Score   : {(1 - avg_fake_prob) * 100:.2f}%")
    print(f"    Frames Analyzed      : {len(records)}")
    print(f"    Peak Manipulation    : {max(fake_probs) * 100:.2f}%")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Video Detection CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, choices=["coatnet", "cnn"], default="coatnet", help="Model type: 'coatnet' or 'cnn'")
    parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights (.pt or .h5)")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    run_prediction(
        video_path=args.video,
        model_type=args.model,
        weights_path=args.weights,
        max_frames=args.frames,
        threshold=args.threshold
    )

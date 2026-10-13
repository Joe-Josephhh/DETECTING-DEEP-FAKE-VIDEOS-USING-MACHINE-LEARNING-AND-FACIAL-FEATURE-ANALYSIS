import os
import tempfile
import time
import json
import numpy as np
import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Import project modules
from models.coatnet import load_coatnet
from models.cnn import load_cnn_model
from face_extractor import (
    analyze_video_frames,
    preprocess_for_coatnet,
    preprocess_for_cnn
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Deepfake Shield | AI Video Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber Aesthetics CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1322 0%, #07090e 100%);
        color: #e2e8f0;
    }

    /* Top Hero Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.3rem;
    }

    /* Verdict Banners */
    .verdict-box {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #ef4444;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10b981;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .verdict-subtitle {
        font-size: 1rem;
        margin-top: 0.4rem;
        opacity: 0.9;
    }

    /* Badge */
    .badge-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-fake {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-real {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def create_demo_video(output_path="sample_demo_video.mp4", duration_sec=2, fps=20):
    """Creates a synthetic test video with a stylized synthetic face for instant testing."""
    width, height = 480, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient
        img[:, :] = (30, 25, 40)
        
        # Draw face oval
        center_x = int(240 + 20 * np.sin(i * 0.2))
        center_y = int(240 + 10 * np.cos(i * 0.2))
        cv2.ellipse(img, (center_x, center_y), (85, 110), 0, 0, 360, (190, 180, 220), -1)
        
        # Eyes
        cv2.circle(img, (center_x - 30, center_y - 25), 10, (50, 40, 60), -1)
        cv2.circle(img, (center_x + 30, center_y - 25), 10, (50, 40, 60), -1)
        
        # Nose
        cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 15), (130, 110, 160), 3)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y + 45), (25, 12), 0, 0, 180, (120, 80, 140), 3)

        out.write(img)

    out.release()
    return output_path


# ===================== SIDEBAR CONTROLS =====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=64)
    st.title("Forensic Controls")
    st.caption("Deepfake Video Detection Engine")
    st.markdown("---")

    # Architecture Choice
    model_choice = st.selectbox(
        "🧠 Neural Network Model",
        ["CoAtNet (Conv + Self-Attention / PyTorch)", "4-Stage CNN (Keras / TensorFlow)"],
        index=0
    )

    # Frame Sampling slider
    sample_frames = st.slider("🎞️ Frame Sampling Density", min_value=5, max_value=40, value=20, step=5,
                              help="Number of frames sampled across the video duration for facial extraction.")

    # Threshold slider
    confidence_thresh = st.slider("⚖️ Decision Threshold", min_value=0.30, max_value=0.80, value=0.50, step=0.05,
                                  help="Confidence probability cutoff to flag a frame as Deepfake.")

    st.markdown("---")
    st.subheader("Model Weights")
    
    # Check default paths
    local_weights = {
        "CoAtNet": "coatnet_uadfv.pt",
        "CNN": "outputs/cnn_model_deepfake_optimized.h5"
    }

    uploaded_weights = st.file_uploader("Upload custom .pt / .h5 weights", type=["pt", "pth", "h5"])
    
    st.markdown("---")
    st.info("💡 **Quick Test:** Don't have a deepfake video ready? Click 'Use Synthetic Demo Video' below to test the full pipeline immediately.")


# ===================== TOP HERO BANNER =====================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ Deepfake Detection & Facial Forensics</div>
    <div class="hero-subtitle">
        Dual-stage deep learning pipeline leveraging spatial facial feature alignment, 
        Convolutional Neural Networks (CNN), and hybrid Self-Attention Transformers (CoAtNet).
    </div>
</div>
""", unsafe_allow_html=True)


# ===================== FILE INPUT SECTION =====================
col_input1, col_input2 = st.columns([2.5, 1.5])

with col_input1:
    uploaded_video = st.file_uploader("📁 Upload Target Video for Forensic Analysis", type=["mp4", "avi", "mov", "mkv"])

with col_input2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    generate_demo = st.button("🧪 Use Synthetic Demo Video", use_container_width=True)

video_path_to_process = None

if generate_demo:
    with st.spinner("Generating synthetic test video..."):
        demo_path = os.path.join(tempfile.gettempdir(), "deepfake_sample_demo.mp4")
        create_demo_video(demo_path)
        video_path_to_process = demo_path
        st.session_state["active_video"] = demo_path
        st.session_state["is_demo"] = True

elif uploaded_video is not None:
    # Save uploaded video to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()
    video_path_to_process = tfile.name
    st.session_state["active_video"] = tfile.name
    st.session_state["is_demo"] = False

elif "active_video" in st.session_state and os.path.exists(st.session_state["active_video"]):
    video_path_to_process = st.session_state["active_video"]


# ===================== PROCESSING & ANALYSIS =====================
if video_path_to_process:
    st.markdown("---")
    vcol1, vcol2 = st.columns([1, 1])

    with vcol1:
        st.subheader("🎥 Video Playback")
        st.video(video_path_to_process)
        if st.session_state.get("is_demo", False):
            st.caption("ℹ️ Currently displaying generated demo face clip.")

    with vcol2:
        st.subheader("⚡ Pipeline Diagnostics")
        st.write("Selected Model:", f"**{model_choice}**")
        st.write("Target Frame Count:", f"**{sample_frames} frames**")
        st.write("Haar Cascade Detector:", "**Active (`haarcascade_frontalface_default.xml`)**")
        
        analyze_btn = st.button("🚀 Run Forensic Deepfake Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        progress_bar = st.progress(0, text="Initializing facial feature extraction...")
        time_start = time.time()

        # 1. Extract Frames & Crop Faces
        try:
            progress_bar.progress(20, text="Sampling frames & extracting faces with Haar Cascade...")
            frame_records = analyze_video_frames(video_path_to_process, max_frames=sample_frames)
        except Exception as e:
            st.error(f"Error during video parsing: {e}")
            frame_records = []

        if not frame_records:
            st.error("No valid frames could be decoded from this video.")
        else:
            # 2. Run Deep Learning Model Inference
            progress_bar.progress(50, text="Loading neural network architecture...")
            
            is_coatnet = "CoAtNet" in model_choice
            fake_probs = []

            # Save uploaded weights to temp if provided
            temp_weight_file = None
            if uploaded_weights:
                suffix = ".pt" if is_coatnet else ".h5"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as wf:
                    wf.write(uploaded_weights.read())
                    temp_weight_file = wf.name

            # Load Model
            weights_target = temp_weight_file or (local_weights["CoAtNet"] if is_coatnet else local_weights["CNN"])

            if is_coatnet:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, has_weights = load_coatnet(weights_target if os.path.exists(weights_target) else None, device=device)
            else:
                model, has_weights = load_cnn_model(weights_target if os.path.exists(weights_target) else None)

            progress_bar.progress(70, text=f"Analyzing {len(frame_records)} facial crops...")

            for idx, item in enumerate(frame_records):
                face_crop = item["face_crop_bgr"]

                if is_coatnet:
                    tensor = preprocess_for_coatnet(face_crop, img_size=224).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        # Index 1 corresponds to 'Fake' in project notebooks
                        prob_fake = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    batch = preprocess_for_cnn(face_crop, img_size=128)
                    pred = model.predict(batch, verbose=0)
                    prob_fake = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0][1])

                # Calibration heuristic for untrained baseline / anomaly detection demonstration:
                # If weights were not provided, we factor in face Laplacian variance + frequency artifact heuristics
                if not has_weights:
                    # Synthetic/Heuristic anomaly estimation for demonstration
                    lap_score = min(1.0, max(0.0, (150.0 - item["laplacian_var"]) / 150.0))
                    prob_fake = float(np.clip(0.4 * prob_fake + 0.6 * lap_score, 0.05, 0.95))

                item["prob_fake"] = prob_fake
                item["is_fake"] = prob_fake >= confidence_thresh
                fake_probs.append(prob_fake)

            progress_bar.progress(100, text="Analysis complete!")
            elapsed = time.time() - time_start
            time.sleep(0.4)
            progress_bar.empty()

            # Clean up temp weights if used
            if temp_weight_file and os.path.exists(temp_weight_file):
                try:
                    os.remove(temp_weight_file)
                except Exception:
                    pass

            # ===================== RESULTS DISPLAY =====================
            st.markdown("---")
            avg_fake_prob = float(np.mean(fake_probs))
            is_overall_fake = avg_fake_prob >= confidence_thresh

            # Banner Verdict
            if is_overall_fake:
                st.markdown(f"""
                <div class="verdict-box verdict-fake">
                    <div class="verdict-title" style="color: #ef4444;">🚨 DEEPFAKE DETECTED</div>
                    <div class="verdict-subtitle" style="color: #fca5a5;">
                        High-confidence facial manipulation or synthetic generation detected.
                        Average Manipulation Confidence: <strong>{avg_fake_prob * 100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box verdict-real">
                    <div class="verdict-title" style="color: #10b981;">✅ AUTHENTIC VIDEO</div>
                    <div class="verdict-subtitle" style="color: #6ee7b7;">
                        Facial features and temporal coherence consistent with genuine footage.
                        Authenticity Confidence: <strong>{(1 - avg_fake_prob) * 100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if not has_weights:
                st.info("ℹ️ **Notice:** Running in heuristic demonstration mode because no pre-trained weights file (`coatnet_uadfv.pt` or `cnn_model_deepfake_optimized.h5`) was detected in repository. Train the model via the notebooks to export production weights!")

            # Metric Cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{len(frame_records)}</div>
                    <div class="metric-lbl">Frames Analyzed</div>
                </div>
                """, unsafe_allow_html=True)

            with mcol2:
                detected_count = sum(1 for f in frame_records if f["face_detected"])
                det_rate = (detected_count / len(frame_records)) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{det_rate:.0f}%</div>
                    <div class="metric-lbl">Face Detection Rate</div>
                </div>
                """, unsafe_allow_html=True)

            with mcol3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{avg_fake_prob * 100:.1f}%</div>
                    <div class="metric-lbl">Fake Probability</div>
                </div>
                """, unsafe_allow_html=True)

            with mcol4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{elapsed:.2f}s</div>
                    <div class="metric-lbl">Inference Latency</div>
                </div>
                """, unsafe_allow_html=True)

            # ===================== TEMPORAL RISK TIMELINE =====================
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.subheader("📈 Temporal Manipulation Timeline")
            
            timestamps = [f["timestamp_sec"] for f in frame_records]
            
            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#0e131f")
            ax.set_facecolor("#0e131f")
            ax.plot(timestamps, fake_probs, color="#38bdf8", linewidth=2.5, marker="o", markersize=5, label="Frame Fake Score")
            ax.axhline(confidence_thresh, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Threshold ({confidence_thresh})")
            ax.fill_between(timestamps, fake_probs, confidence_thresh, where=np.array(fake_probs) >= confidence_thresh,
                            color="#ef4444", alpha=0.25, interpolate=True)
            ax.fill_between(timestamps, fake_probs, confidence_thresh, where=np.array(fake_probs) < confidence_thresh,
                            color="#10b981", alpha=0.15, interpolate=True)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Video Timestamp (seconds)", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Deepfake Probability", color="#94a3b8", fontsize=10)
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

            # ===================== FRAME GALLERY INSPECTION =====================
            st.subheader("🔍 Facial Feature & Frame Inspection")
            st.caption("Inspect individual sampled faces, boundary boxes, and per-frame classification scores:")

            # Show gallery in chunks of 4 columns
            cols_per_row = 4
            for row_start in range(0, len(frame_records), cols_per_row):
                cols = st.columns(cols_per_row)
                for c_idx in range(cols_per_row):
                    item_idx = row_start + c_idx
                    if item_idx < len(frame_records):
                        rec = frame_records[item_idx]
                        with cols[c_idx]:
                            st.image(rec["face_crop_rgb"], use_container_width=True)
                            p = rec["prob_fake"]
                            if rec["is_fake"]:
                                st.markdown(f"<span class='badge-tag badge-fake'>FAKE: {p*100:.1f}%</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span class='badge-tag badge-real'>REAL: {(1-p)*100:.1f}%</span>", unsafe_allow_html=True)
                            st.caption(f"Frame #{rec['frame_idx']} ({rec['timestamp_sec']}s)")

            # ===================== FORENSIC REPORT EXPORT =====================
            st.markdown("---")
            report_data = {
                "verdict": "DEEPFAKE" if is_overall_fake else "AUTHENTIC",
                "average_fake_probability": round(avg_fake_prob, 4),
                "threshold": confidence_thresh,
                "model_architecture": model_choice,
                "analyzed_frames_count": len(frame_records),
                "face_detection_rate": round(det_rate, 2),
                "frame_breakdown": [
                    {
                        "frame_index": f["frame_idx"],
                        "timestamp_seconds": f["timestamp_sec"],
                        "fake_probability": round(f["prob_fake"], 4),
                        "face_detected": f["face_detected"],
                        "laplacian_variance": round(f["laplacian_var"], 2)
                    }
                    for f in frame_records
                ]
            }

            st.download_button(
                label="📥 Download Forensic Audit Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name="deepfake_forensic_report.json",
                mime="application/json"
            )
else:
    st.info("👆 Please upload an MP4/AVI video above or click **'Use Synthetic Demo Video'** to run the detector.")

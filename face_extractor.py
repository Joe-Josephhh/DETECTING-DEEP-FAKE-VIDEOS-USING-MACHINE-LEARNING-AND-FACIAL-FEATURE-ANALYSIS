import os
import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_detector = None

def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    return _face_detector


def extract_face_from_frame(frame, face_detector=None, margin=0.1):
    """
    Detects face using Haar Cascade. If no face is detected,
    falls back to center crop of the frame (as in project notebooks).
    Returns (cropped_face_bgr, bbox_tuple, is_detected).
    """
    if face_detector is None:
        face_detector = get_face_detector()

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Choose the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        # Add optional margin
        mx = int(fw * margin)
        my = int(fh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        face_crop = frame[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2 - x1, y2 - y1), True
    else:
        # Center fallback crop
        startx = w // 4
        starty = h // 4
        face_crop = frame[starty:starty + h // 2, startx:startx + w // 2]
        return face_crop, (startx, starty, w // 2, h // 2), False


def preprocess_for_cnn(face_bgr, img_size=128):
    """
    Keras CNN Preprocessing matching cnn_udafv_new.ipynb:
    - Resize to (img_size, img_size)
    - GaussianBlur (3, 3)
    - BGR -> RGB -> HSV
    - Histogram Equalization on Value channel
    - Normalize to [0, 1]
    """
    face_resized = cv2.resize(face_bgr, (img_size, img_size))
    face_blur = cv2.GaussianBlur(face_resized, (3, 3), 0)
    face_rgb = cv2.cvtColor(face_blur, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    face_norm = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
    return np.expand_dims(face_norm.astype(np.float32), axis=0)


def preprocess_for_coatnet(face_bgr, img_size=224):
    """
    PyTorch CoAtNet Preprocessing matching coatnet_uadfv_pipeline.ipynb:
    - Resize to (224, 224)
    - Normalize with ImageNet mean and std
    - Returns torch.Tensor of shape (1, 3, 224, 224)
    """
    import torch
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (img_size, img_size))
    arr = face_resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def analyze_video_frames(video_path, max_frames=20):
    """
    Samples up to max_frames evenly spaced from the video.
    Returns a list of dicts:
    [{
        'frame_idx': int,
        'timestamp_sec': float,
        'original_bgr': np.ndarray,
        'face_crop_bgr': np.ndarray,
        'face_crop_rgb': np.ndarray,
        'bbox': (x, y, w, h),
        'face_detected': bool,
        'laplacian_var': float, # sharpness / boundary artifact indicator
    }]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        total_frames = 100

    step = max(1, total_frames // max_frames)
    frame_indices = [i * step for i in range(min(max_frames, total_frames))]

    detector = get_face_detector()
    extracted_data = []

    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        face_crop, bbox, is_detected = extract_face_from_frame(frame, detector)
        
        # Calculate Laplacian variance for facial blur/artifact analysis
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box on a copy of original frame for preview
        annotated_frame = frame.copy()
        if bbox:
            x, y, w, h = bbox
            color = (0, 255, 0) if is_detected else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label = "Face" if is_detected else "Center-Crop"
            cv2.putText(annotated_frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        extracted_data.append({
            "frame_idx": target_idx,
            "timestamp_sec": round(target_idx / fps, 2),
            "original_annotated_rgb": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            "face_crop_bgr": face_crop,
            "face_crop_rgb": face_rgb,
            "bbox": bbox,
            "face_detected": is_detected,
            "laplacian_var": lap_var
        })

    cap.release()
    return extracted_data

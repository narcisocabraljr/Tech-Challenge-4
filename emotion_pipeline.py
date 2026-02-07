import cv2
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from tqdm import tqdm
import glob
import os

# =========================
# CONFIGURAÇÕES
# =========================
VIDEO_PATH = "videos/sample.mp4"
FRAME_SKIP = 10  # processa 1 a cada N frames
CONFIDENCE_THRESHOLD = 0.5

# =========================
# MODELOS
# =========================
yolo = YOLO("yolov8n.pt")  # modelo leve

# =========================
# FUNÇÕES
# =========================
def analyze_face(face_img):
    try:
        result = DeepFace.analyze(
            face_img,
            actions=["emotion"],
            enforce_detection=False
        )
        return result[0]["dominant_emotion"]
    except Exception:
        return None


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frame_id = 0

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        detections = yolo(frame)[0]

        for box in detections.boxes:
            if box.conf < CONFIDENCE_THRESHOLD:
                continue

            cls = int(box.cls[0])
            label = yolo.names[cls]

            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            emotion = analyze_face(face)

            results.append({
                "frame": frame_id,
                "emotion": emotion
            })

    cap.release()
    return results


def generate_report(data):
    df = pd.DataFrame(data)
    summary = df["emotion"].value_counts(normalize=True) * 100
    summary_df = summary.reset_index()
    summary_df.columns = ["emotion", "percentage"]
    return df, summary_df

def download_videos():
    video_list = glob.glob("RAVDESS/Video_Speech_Actor_*/*.mp4")
    print(len(video_list))
    print(video_list[:3])
    return video_list[:10]  # limita para pós
# =========================
# MAIN
# =========================
if __name__ == "__main__":
    downloaded_videos = download_videos()
    print("Processando vídeos...")
    for video in downloaded_videos:
        print(f"Processando vídeo: {video}")
        raw_data = process_video(video)
        df_frames, df_summary = generate_report(raw_data)
    df_frames.to_csv("outputs/frame_emotions.csv", index=False)
    df_summary.to_csv("outputs/emotion_summary.csv", index=False)

    print("\nResumo emocional (%):")
    print(df_summary)

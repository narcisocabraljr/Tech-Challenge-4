import os
import glob

# Suppress TensorFlow info/warning logs (must be done before importing deepface/tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deepface import DeepFace
from tqdm import tqdm
from audio_emotion_analyzer import AudioEmotionAnalyzer

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

def plot_emotion_timeline(multimodal_results, video_name, output_dir="outputs"):
    df = pd.DataFrame(multimodal_results)

    # Converte emoções para números para plotar
    emotion_map = {e: i for i, e in enumerate(df["combined_emotion"].unique())}
    df["emotion_code"] = df["combined_emotion"].map(emotion_map)

    plt.figure(figsize=(12, 4))
    plt.plot(df["frame"], df["emotion_code"], marker='o')
    plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
    plt.xlabel("Frame")
    plt.ylabel("Emoção")
    plt.title(f"Timeline Emocional - {video_name}")
    plt.grid(True)

    output_path = os.path.join(output_dir, f"{video_name}_timeline.png")
    plt.savefig(output_path)
    plt.close()

    print(f"  📊 Timeline salva em: {output_path}")

def generate_report(data):
    df = pd.DataFrame(data)
    summary = df["emotion"].value_counts(normalize=True) * 100
    summary_df = summary.reset_index()
    summary_df.columns = ["emotion", "percentage"]
    return df, summary_df

def download_videos():
    video_list = glob.glob("RAVDESS/Video_Speech_Actor_*/*.mp4")
    print(f"Total de vídeos encontrados: {len(video_list)}")
    print(f"Primeiros vídeos: {video_list[:3]}")
    # Processa primeiros 20 vídeos (ajuste conforme necessário)
    selected = video_list[:20]
    print(f"\n📹 Selecionados para processamento: {len(selected)} vídeos\n")
    return selected


def combine_audio_visual_emotions(visual_emotion, audio_emotion, audio_confidence=0.3):
    """
    Combina emoções detectadas por áudio e vídeo usando fusão multimodal.
    
    Args:
        visual_emotion: Emoção detectada pela análise facial
        audio_emotion: Emoção detectada pela análise de áudio
        audio_confidence: Peso da emoção de áudio (0-1)
        
    Returns:
        Emoção combinada
    """
    # Se ambas são iguais, retorna com confiança alta
    if visual_emotion == audio_emotion:
        return visual_emotion, "high_confidence"
    
    # Se uma delas é None, retorna a outra
    if visual_emotion is None:
        return audio_emotion, "audio_only"
    if audio_emotion is None:
        return visual_emotion, "visual_only"
    
    # Se áudio é neutral mas visual não é, prioriza visual
    # (áudio neutral é menos informativo)
    if audio_emotion == "neutral" and visual_emotion != "neutral":
        return visual_emotion, "visual_priority"
    
    # Se visual é neutral mas áudio não é, prioriza áudio
    if visual_emotion == "neutral" and audio_emotion != "neutral":
        return audio_emotion, "audio_priority"
    
    # Em caso de conflito entre emoções não-neutras
    # Prioriza VISUAL (DeepFace é mais confiável que heurísticas)
    return visual_emotion, "visual_priority"


def process_multimodal_video(video_path, audio_analyzer):
    """
    Processa vídeo com análise multimodal (áudio + visual).
    
    Args:
        video_path: Caminho do vídeo
        audio_analyzer: Instância do AudioEmotionAnalyzer
        
    Returns:
        Resultados combinados
    """
    # Análise visual (frames)
    visual_results = process_video(video_path)
    
    # Análise de áudio com garantia de limpeza
    audio_path = None
    audio_result = None
    
    try:
        audio_path = audio_analyzer.extract_audio_from_video(video_path)
        
        if audio_path:
            audio_result = audio_analyzer.process_audio(audio_path)
        else:
            print(f"  ⚠️  Falha na extração de áudio para: {os.path.basename(video_path)}")
    finally:
        # Garante que o arquivo temporário seja deletado mesmo em caso de erro
        if audio_path and os.path.exists(audio_path) and "temp_" in audio_path:
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"  ⚠️  Aviso: Não foi possível deletar {audio_path}: {e}")
    
    # Combina resultados
    multimodal_results = []
    
    for visual_data in visual_results:
        audio_emotion = audio_result['emotion'] if audio_result else None
        visual_emotion = visual_data['emotion']
        
        combined_emotion, confidence = combine_audio_visual_emotions(
            visual_emotion, 
            audio_emotion
        )
        
        multimodal_results.append({
            "frame": visual_data['frame'],
            "visual_emotion": visual_emotion,
            "audio_emotion": audio_emotion,
            "combined_emotion": combined_emotion,
            "confidence": confidence
        })
    
    # Adiciona informações de áudio ao resultado
    audio_info = {
        "audio_emotion": audio_result['emotion'] if audio_result else None,
        "is_anomaly": audio_result['anomaly_detection']['is_anomaly'] if audio_result else False,
        "emotional_stability": audio_result['emotional_variation']['stability'] if audio_result else "unknown",
        "audio_features": audio_result['features'] if audio_result else None
    }
    
    return multimodal_results, audio_info


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    downloaded_videos = download_videos()
    audio_analyzer = AudioEmotionAnalyzer()
    
    # Verifica FFmpeg antes de começar
    if not audio_analyzer.check_ffmpeg():
        print("\n" + "!"*60)
        print("❌ ERRO CRÍTICO: FFmpeg não encontrado no sistema!")
        print("   A análise de áudio não funcionará. Instale o FFmpeg.")
        print("!"*60 + "\n")

    output_dir = "outputs"
    all_results = []
    all_audio_info = []
    errors = []  # Para rastrear erros
    
    print("Processando vídeos com análise multimodal (áudio + visual)...")
    for video in tqdm(downloaded_videos, desc="Vídeos"):
        video_name = os.path.basename(video)
        print(f"\nProcessando: {video_name}")
        
        try:
            # Processa com análise multimodal
            multimodal_results, audio_info = process_multimodal_video(video, audio_analyzer)
            
            # Verifica se o processamento retornou resultados válidos
            if not multimodal_results:
                raise ValueError("Nenhum frame processado (vídeo vazio ou sem faces detectadas)")
            
            # Adiciona nome do vídeo aos resultados
            for result in multimodal_results:
                result['video'] = video_name
            
            all_results.extend(multimodal_results)
            
            # Adiciona informações de áudio
            audio_info['video'] = video_name
            all_audio_info.append(audio_info)
            
            print(f"  ✓ Emoção de áudio: {audio_info['audio_emotion']}")
            print(f"  ✓ Anomalia detectada: {audio_info['is_anomaly']}")
            print(f"  ✓ Estabilidade emocional: {audio_info['emotional_stability']}")
            print(f"  ✓ Frames analisados: {len(multimodal_results)}")
            plot_emotion_timeline(multimodal_results, video_name, output_dir)
            
        except Exception as e:
            error_msg = f"{video_name}: {type(e).__name__}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ ERRO: {error_msg}")
            # Ainda adiciona resultado visual se houver
            continue
    
    # Gera relatórios
    if all_results:
        df_multimodal = pd.DataFrame(all_results)
        df_audio_summary = pd.DataFrame(all_audio_info)
        
        # Salva resultados detalhados (sep=';' para compatibilidade com Excel pt-BR)
        df_multimodal.to_csv("outputs/multimodal_emotions.csv", index=False, sep=';')
        df_audio_summary.to_csv("outputs/audio_analysis_summary.csv", index=False, sep=';')
        
        # Gera resumo estatístico
        print("\n" + "="*50)
        print("RESUMO DA ANÁLISE MULTIMODAL")
        print("="*50)
        
        print("\n📊 Distribuição de Emoções Combinadas:")
        combined_summary = df_multimodal["combined_emotion"].value_counts(normalize=True) * 100
        for emotion, pct in combined_summary.items():
            print(f"  {emotion}: {pct:.2f}%")
        
        print("\n🎵 Análise de Áudio:")
        audio_emotion_summary = df_audio_summary["audio_emotion"].value_counts(normalize=True) * 100
        for emotion, pct in audio_emotion_summary.items():
            print(f"  {emotion}: {pct:.2f}%")
        
        print("\n⚠️ Anomalias Detectadas:")
        anomaly_count = df_audio_summary["is_anomaly"].sum()
        total_videos = len(df_audio_summary)
        print(f"  {anomaly_count} de {total_videos} vídeos ({anomaly_count/total_videos*100:.1f}%)")
        
        print("\n💚 Estabilidade Emocional:")
        stability_summary = df_audio_summary["emotional_stability"].value_counts()
        for stability, count in stability_summary.items():
            print(f"  {stability}: {count} vídeos")
        
        print("\n✅ Relatórios salvos em:")
        print("  - outputs/multimodal_emotions.csv")
        print("  - outputs/audio_analysis_summary.csv")
    else:
        print("\n⚠️ Nenhum resultado para processar.")
    
    # Relatório de erros
    if errors:
        print("\n" + "="*80)
        print(f"❌ RELATÓRIO DE ERROS ({len(errors)} vídeos falharam):")
        print("="*80)
        for error_msg in errors:
            print(f"  • {error_msg}")
        
        # Salvar erros em arquivo de log
        log_file = os.path.join(output_dir, "processing_errors.log")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Relatório de Erros - {len(errors)} vídeos falharam\n")
            f.write("="*80 + "\n\n")
            for error_msg in errors:
                f.write(f"{error_msg}\n\n")
        print(f"\n📋 Log de erros salvo em: {log_file}")
    else:
        print("\n✅ Todos os vídeos foram processados com sucesso!")

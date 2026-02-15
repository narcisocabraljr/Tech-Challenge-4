import librosa
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import soundfile as sf
import subprocess
import os
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MFCC = 13


class AudioEmotionAnalyzer:
    """
    Analisador de emoções baseado em características acústicas de áudio.
    Extrai features como MFCCs, pitch, energia, ritmo e pausas para
    detectar emoções e anomalias no estado emocional.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_history = []
        
    @staticmethod
    def check_ffmpeg():
        """Verifica se o FFmpeg está instalado e acessível."""
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False

    def extract_audio_from_video(self, video_path, output_audio_path=None):
        """
        Extrai a trilha de áudio de um vídeo usando ffmpeg.
        
        Args:
            video_path: Caminho do arquivo de vídeo
            output_audio_path: Caminho de saída do áudio (opcional)
            
        Returns:
            Caminho do arquivo de áudio extraído
        """
        if output_audio_path is None:
            video_name = Path(video_path).stem
            output_audio_path = f"temp_{video_name}.wav"
        
        try:
            # Extrai áudio usando ffmpeg
            command = [
                'ffmpeg', '-i', video_path,
                '-vn',  # Sem vídeo
                '-acodec', 'pcm_s16le',  # Codec PCM
                '-ar', str(SAMPLE_RATE),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Sobrescrever
                output_audio_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            return output_audio_path
        except FileNotFoundError:
            print(f"ERRO: FFmpeg não encontrado. Instale o FFmpeg para habilitar a análise de áudio.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Erro ao extrair áudio: {e}")
            if e.stderr:
                print(f"Detalhes FFmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
            return None
    
    def extract_acoustic_features(self, audio_path):
        """
        Extrai características acústicas completas do áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            Dicionário com todas as features extraídas
        """
        # Carrega o áudio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        features = {}
        
        # 1. MFCCs (Mel-frequency cepstral coefficients) - Timbre/Tom
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        features['mfcc_min'] = np.min(mfccs, axis=1)
        features['mfcc_max'] = np.max(mfccs, axis=1)
        
        # 2. Pitch (F0) - Frequência fundamental
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=HOP_LENGTH)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
        
        # 3. Energia/Intensidade - Volume
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_min'] = np.min(rms)
        features['energy_max'] = np.max(rms)
        
        # 4. Zero Crossing Rate - Ritmo e textura
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. Spectral Features - Características espectrais
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # 6. Tempo - Velocidade da fala/ritmo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0
        
        # 7. Pausas e Silêncios
        intervals = librosa.effects.split(y, top_db=20)
        if len(intervals) > 0:
            pause_durations = []
            for i in range(len(intervals) - 1):
                pause_duration = (intervals[i+1][0] - intervals[i][1]) / sr
                pause_durations.append(pause_duration)
            
            if pause_durations:
                features['pause_mean'] = np.mean(pause_durations)
                features['pause_std'] = np.std(pause_durations)
                features['pause_count'] = len(pause_durations)
            else:
                features['pause_mean'] = 0
                features['pause_std'] = 0
                features['pause_count'] = 0
        else:
            features['pause_mean'] = 0
            features['pause_std'] = 0
            features['pause_count'] = 0
        
        # 8. Chroma Features - Características tonais
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def classify_emotion_from_features(self, features):
        """
        Classifica emoção baseada em características acústicas.
        Usa regras heurísticas baseadas em pesquisas de Speech Emotion Recognition.
        
        Args:
            features: Dicionário com features extraídas
            
        Returns:
            Emoção detectada (string)
        """
        # Regras heurísticas baseadas em literatura de SER
        
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        energy_mean = features.get('energy_mean', 0)
        energy_std = features.get('energy_std', 0)
        tempo = features.get('tempo', 0)
        zcr_mean = features.get('zcr_mean', 0)
        
        # Normaliza tempo (pode vir como array)
        if hasattr(tempo, '__iter__'):
            tempo = tempo[0] if len(tempo) > 0 else 0
        
        # Classificação baseada em padrões conhecidos (thresholds refinados)
        # Ordem de prioridade: condições mais específicas primeiro
        
        # 1. Tristeza: Baixa energia + pitch baixo + ritmo lento (mais específico)
        if energy_mean < 0.003 and pitch_mean < 250 and tempo < 100:
            return "sad"
        
        # 2. Raiva: MUITO alta variação de pitch + ritmo elevado + energia moderada
        elif pitch_std > 750 and tempo > 125 and energy_mean > 0.004:
            return "angry"
        
        # 3. Medo: Alta variação de pitch + ZCR elevado + energia baixa/moderada
        elif pitch_std > 550 and zcr_mean > 0.075 and energy_mean < 0.007:
            return "fear"
        
        # 4. Surpresa: Variação de energia significativa + pitch variável
        elif energy_std > 0.004 and pitch_std > 350 and pitch_std < 600:
            return "surprise"
        
        # 5. Felicidade: Pitch alto + ritmo moderado/alto + energia moderada
        elif pitch_mean > 280 and tempo > 110 and energy_mean > 0.004:
            return "happy"
        
        # 6. Neutro: Qualquer outro padrão (mais estável ou sem características marcantes)
        else:
            return "neutral"
    
    def detect_anomalies(self, features_list, current_features, threshold=2.0):
        """
        Detecta anomalias comparando features atuais com histórico.
        Usa Z-score para identificar variações significativas.
        
        Args:
            features_list: Lista de features históricas
            current_features: Features atuais
            threshold: Limite do Z-score para considerar anomalia (padrão: 2.0)
            
        Returns:
            Dicionário com anomalias detectadas
        """
        if len(features_list) < 3:
            return {"is_anomaly": False, "anomalies": [], "z_scores": {}}
        
        # Converte para DataFrame para facilitar cálculos
        df_history = pd.DataFrame(features_list)
        
        anomalies = []
        z_scores = {}
        
        # Verifica features-chave para anomalias
        key_features = ['pitch_mean', 'energy_mean', 'tempo', 'mfcc_mean']
        
        for feature in key_features:
            if feature in current_features and feature in df_history.columns:
                # Calcula Z-score
                if feature == 'mfcc_mean':
                    # Para arrays, usa a média
                    current_val = np.mean(current_features[feature])
                    hist_vals = df_history[feature].apply(np.mean)
                else:
                    current_val = current_features[feature]
                    hist_vals = df_history[feature]
                
                mean = hist_vals.mean()
                std = hist_vals.std()
                
                if std > 0:
                    z_score = (current_val - mean) / std
                    z_scores[feature] = z_score
                    
                    if abs(z_score) > threshold:
                        anomalies.append({
                            "feature": feature,
                            "z_score": z_score,
                            "current_value": current_val,
                            "expected_mean": mean,
                            "deviation": "high" if z_score > 0 else "low"
                        })
        
        return {
            "is_anomaly": len(anomalies) > 0,
            "anomalies": anomalies,
            "z_scores": z_scores,
            "anomaly_count": len(anomalies)
        }
    
    def _extract_scalar(self, value):
        """
        Extrai um valor escalar de qualquer tipo de dado.
        
        Args:
            value: Valor que pode ser escalar, array, lista, etc.
            
        Returns:
            float: Valor escalar extraído
        """
        # Se já é escalar
        if np.isscalar(value):
            return float(value)
        
        # Se é array numpy
        if isinstance(value, np.ndarray):
            # Achatar completamente e pegar a média
            flat = value.flatten()
            if len(flat) > 0:
                return float(np.mean(flat))
            return 0.0
        
        # Se é lista ou tupla
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                # Recursivamente extrai escalar do primeiro elemento
                return self._extract_scalar(value[0])
            return 0.0
        
        # Fallback: tentar converter diretamente
        try:
            return float(value)
        except:
            return 0.0
    
    def analyze_emotional_variations(self, features_list):
        """
        Analisa variações no estado emocional ao longo do tempo.
        
        Args:
            features_list: Lista de features ao longo do tempo
            
        Returns:
            Análise estatística das variações
        """
        if len(features_list) < 2:
            return {"stability": "insufficient_data"}
        
        # Normaliza features para extrair apenas valores escalares
        normalized_features = []
        for features in features_list:
            normalized = {}
            for key, value in features.items():
                # Skip MFCCs complexos
                if key in ['mfcc_mean', 'mfcc_std', 'mfcc_min', 'mfcc_max']:
                    continue
                # Extrai escalar de qualquer tipo de valor
                normalized[key] = self._extract_scalar(value)
            normalized_features.append(normalized)
        
        df = pd.DataFrame(normalized_features)
        
        # Calcula variação para features-chave
        variations = {}
        
        for col in ['pitch_mean', 'energy_mean', 'tempo']:
            if col in df.columns:
                values = df[col].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Coeficiente de variação (CV)
                cv = std_val / mean_val if mean_val > 0 else 0
                
                variations[col] = {
                    "coefficient_of_variation": float(cv),
                    "std": float(std_val),
                    "mean": float(mean_val),
                    "trend": "increasing" if values[-1] > values[0] else "decreasing"
                }
        
        # Determina estabilidade emocional geral
        if variations:
            avg_cv = np.mean([v['coefficient_of_variation'] for v in variations.values()])
        else:
            avg_cv = 0.0
        
        if avg_cv < 0.2:
            stability = "stable"
        elif avg_cv < 0.5:
            stability = "moderate"
        else:
            stability = "unstable"
        
        return {
            "stability": stability,
            "average_variation": float(avg_cv),
            "feature_variations": variations
        }
    
    def process_audio(self, audio_path):
        """
        Processa um arquivo de áudio completo e retorna análise emocional.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            Dicionário com análise completa
        """
        # Reset features_history para este novo áudio
        self.features_history = []
        
        # Extrai features
        features = self.extract_acoustic_features(audio_path)
        
        # Classifica emoção
        emotion = self.classify_emotion_from_features(features)
        
        # Detecta anomalias (se houver histórico)
        anomaly_info = self.detect_anomalies(self.features_history, features)
        
        # Adiciona ao histórico
        self.features_history.append(features)
        
        # Analisa variações
        variation_info = self.analyze_emotional_variations(self.features_history)
        
        return {
            "emotion": emotion,
            "features": features,
            "anomaly_detection": anomaly_info,
            "emotional_variation": variation_info
        }


def process_video_with_audio(video_path):
    """
    Processa um vídeo extraindo e analisando o áudio.
    
    Args:
        video_path: Caminho do vídeo
        
    Returns:
        Análise emocional do áudio
    """
    analyzer = AudioEmotionAnalyzer()
    audio_path = None
    result = None
    
    try:
        # Extrai áudio
        audio_path = analyzer.extract_audio_from_video(video_path)
        
        if audio_path is None:
            return None
        
        # Analisa áudio
        result = analyzer.process_audio(audio_path)
    finally:
        # Garante que o arquivo temporário seja deletado mesmo em caso de erro
        if audio_path and os.path.exists(audio_path) and "temp_" in audio_path:
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Aviso: Não foi possível deletar {audio_path}: {e}")
    
    return result


# =========================
# TESTE STANDALONE
# =========================
if __name__ == "__main__":
    # Teste com um vídeo exemplo
    import glob
    
    videos = glob.glob("RAVDESS/Video_Speech_Actor_*/*.mp4")
    
    if videos:
        print(f"Testando com: {videos[0]}")
        result = process_video_with_audio(videos[0])
        
        if result:
            print(f"\nEmoção detectada: {result['emotion']}")
            print(f"\nAnomalia detectada: {result['anomaly_detection']['is_anomaly']}")
            print(f"Estabilidade emocional: {result['emotional_variation']['stability']}")
            
            print("\n=== Features Principais ===")
            features = result['features']
            print(f"Pitch médio: {features['pitch_mean']:.2f} Hz")
            print(f"Energia média: {features['energy_mean']:.4f}")
            print(f"Tempo: {features['tempo']:.2f} BPM")
            print(f"Pausas detectadas: {features['pause_count']}")

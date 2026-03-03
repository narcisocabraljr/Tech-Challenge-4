"""
Módulo de Fusão Multimodal Avançada - Tech Challenge 4
=======================================================
Combina emoções detectadas por análise visual (DeepFace) e áudio (Librosa)
utilizando fusão adaptativa com detecção de incongruência e pesos dinâmicos
baseados na qualidade de cada modalidade.

Melhorias em relação à versão anterior:
- Pesos adaptativos (não mais fixos 0.5/0.5)
- Detecção de incongruência emocional (clinicamente relevante)
- Score de qualidade por modalidade
- Métricas detalhadas exportadas para o CSV
"""

import numpy as np
from typing import Optional, Tuple

# Mapa de valência emocional: emoções positivas, negativas e neutras
EMOTION_VALENCE = {
    "happy": "positive",
    "surprise": "neutral",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fear": "negative",
    "disgust": "negative",
}

# Intensidade emocional (arousal): alta, média, baixa
EMOTION_AROUSAL = {
    "angry": "high",
    "fear": "high",
    "surprise": "high",
    "happy": "medium",
    "sad": "low",
    "neutral": "low",
    "disgust": "medium",
}

# Emoções opostas — alta incongruência se ocorrerem simultaneamente
OPPOSITE_EMOTIONS = {
    ("happy", "sad"),
    ("angry", "fear"),
    ("happy", "angry"),
    ("happy", "disgust"),
}


def calculate_incongruence_score(visual_emotion: Optional[str], audio_emotion: Optional[str]) -> float:
    """
    Calcula um score de incongruência (0.0 – 1.0) entre as modalidades.

    Regras:
    - Mesmo emotion → 0.0
    - Emoções na mesma valência → 0.2
    - Uma neutra, outra intensa → 0.3
    - Valências opostas → 0.7
    - Emoções diametralmente opostas → 0.9

    Args:
        visual_emotion: Emoção detectada pela análise facial
        audio_emotion: Emoção detectada pela análise de áudio

    Returns:
        Score de incongruência (0.0 = concordância total, 1.0 = máxima incongruência)
    """
    if visual_emotion is None or audio_emotion is None:
        return 0.0

    if visual_emotion == audio_emotion:
        return 0.0

    # Par canônico ordenado
    pair = tuple(sorted([visual_emotion, audio_emotion]))
    if pair in {tuple(sorted(p)) for p in OPPOSITE_EMOTIONS}:
        return 0.9

    v_val = EMOTION_VALENCE.get(visual_emotion, "neutral")
    a_val = EMOTION_VALENCE.get(audio_emotion, "neutral")

    v_ar = EMOTION_AROUSAL.get(visual_emotion, "low")
    a_ar = EMOTION_AROUSAL.get(audio_emotion, "low")

    score = 0.0

    # Valências opostas (positivo vs negativo)
    if {v_val, a_val} == {"positive", "negative"}:
        score += 0.7
    elif "neutral" in {v_val, a_val} and {v_val, a_val} != {"neutral", "neutral"}:
        score += 0.3
    elif v_val == a_val:
        score += 0.1  # Mesma valência mas emoções diferentes
    else:
        score += 0.4

    # Arousal divergente aumenta incongruência
    arousal_order = {"low": 0, "medium": 1, "high": 2}
    ar_diff = abs(arousal_order.get(v_ar, 0) - arousal_order.get(a_ar, 0))
    score += ar_diff * 0.1

    return min(round(score, 3), 1.0)


def assess_visual_quality(detection_confidence: float, has_face: bool) -> float:
    """
    Estima a qualidade da análise visual (0.0 – 1.0).

    Args:
        detection_confidence: Confiança do detector YOLO (0-1)
        has_face: Se o DeepFace detectou face com sucesso

    Returns:
        Score de qualidade visual
    """
    if not has_face:
        return 0.1
    return min(0.4 + detection_confidence * 0.6, 1.0)


def assess_audio_quality(features: Optional[dict]) -> float:
    """
    Estima a qualidade da análise de áudio (0.0 – 1.0).

    Args:
        features: Dicionário de features acústicas

    Returns:
        Score de qualidade do áudio
    """
    if features is None:
        return 0.0

    score = 0.5  # Base

    energy = features.get("energy_mean", 0)
    if energy > 0.001:
        score += 0.2
    if energy > 0.003:
        score += 0.1

    pitch = features.get("pitch_mean", 0)
    if pitch > 50:
        score += 0.1

    pause_count = features.get("pause_count", 0)
    if pause_count > 0:
        score += 0.1

    return min(round(score, 3), 1.0)


def fuse_emotions_advanced(
    visual_emotion: Optional[str],
    audio_emotion: Optional[str],
    visual_quality: float = 0.7,
    audio_quality: float = 0.7,
) -> Tuple[str, str, float, float, float]:
    """
    Fusão multimodal avançada com pesos dinâmicos.

    Args:
        visual_emotion: Emoção detectada visualmente
        audio_emotion: Emoção detectada pelo áudio
        visual_quality: Qualidade da modalidade visual (0-1)
        audio_quality: Qualidade da modalidade de áudio (0-1)

    Returns:
        Tupla: (emoção_combinada, confidence_label, incongruence_score,
                visual_weight, audio_weight)
    """
    incongruence = calculate_incongruence_score(visual_emotion, audio_emotion)

    # --- Casos sem dados ---
    if visual_emotion is None and audio_emotion is None:
        return "unknown", "no_data", incongruence, 0.0, 0.0

    if visual_emotion is None:
        return audio_emotion, "audio_only", incongruence, 0.0, 1.0

    if audio_emotion is None:
        return visual_emotion, "visual_only", incongruence, 1.0, 0.0

    # --- Concordância perfeita ---
    if visual_emotion == audio_emotion:
        # Pesos balanceados quando há concordância
        total_q = visual_quality + audio_quality
        vw = visual_quality / total_q if total_q > 0 else 0.5
        aw = audio_quality / total_q if total_q > 0 else 0.5
        return visual_emotion, "high_confidence", incongruence, round(vw, 3), round(aw, 3)

    # --- Calcular pesos adaptativos ---
    total_q = visual_quality + audio_quality
    if total_q == 0:
        visual_weight = 0.5
        audio_weight = 0.5
    else:
        visual_weight = visual_quality / total_q
        audio_weight = audio_quality / total_q

    # --- Regras de fusão por qualidade ---

    # Se uma modalidade tem qualidade muito baixa, priorizar a outra
    if visual_quality < 0.3 and audio_quality >= 0.5:
        return audio_emotion, "audio_priority", incongruence, round(visual_weight, 3), round(audio_weight, 3)

    if audio_quality < 0.3 and visual_quality >= 0.5:
        return visual_emotion, "visual_priority", incongruence, round(visual_weight, 3), round(audio_weight, 3)

    # Regras semânticas (quando qualidade é similar)
    if audio_emotion == "neutral" and visual_emotion != "neutral":
        # Áudio neutro = pouco informativo; priorizar expressão facial
        return visual_emotion, "visual_priority", incongruence, round(visual_weight + 0.2, 3), round(audio_weight - 0.2, 3)

    if visual_emotion == "neutral" and audio_emotion != "neutral":
        # Face neutra = rosto inexpressivo; priorizar voz
        return audio_emotion, "audio_priority", incongruence, round(visual_weight - 0.2, 3), round(audio_weight + 0.2, 3)

    # Conflito entre emoções não-neutras: usar pesos e qualidade para decidir
    if visual_weight >= audio_weight:
        return visual_emotion, "visual_priority", incongruence, round(visual_weight, 3), round(audio_weight, 3)
    else:
        return audio_emotion, "audio_priority", incongruence, round(visual_weight, 3), round(audio_weight, 3)

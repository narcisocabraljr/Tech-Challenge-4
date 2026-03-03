"""
Módulo de Análise Clínica - Tech Challenge 4
=============================================
Fornece contextualização médica para os resultados de análise emocional
multimodal. Detecta padrões associados a condições clínicas como depressão,
ansiedade e agitação, gerando alertas e scores de risco.

Referências Científicas:
- El Ayadi et al. (2011). Survey on speech emotion recognition: Features,
  classification schemes and databases. Pattern Recognition, 44(3), 572–587.
- Schuller, B. et al. (2013). The INTERSPEECH 2013 Computational Paralinguistics
  Challenge. Interspeech, 148–152.
- WHO (2022). World Mental Health Report. World Health Organization.
- DSM-5-TR: Diagnostic and Statistical Manual of Mental Disorders, 5th Ed.
"""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# Carrega thresholds do arquivo de configuração
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "medical_thresholds.yaml")

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback com valores padrão se YAML não for encontrado
        return {}

CONFIG = _load_config()


# ============================================================
# CLASSES DE DOMÍNIO MÉDICO
# ============================================================

class EmotionProfile:
    """Perfil emocional extraído de um segmento de vídeo/sessão."""

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self._compute_distributions()

    def _compute_distributions(self):
        total = len(self.df)
        if total == 0:
            self.visual_dist = {}
            self.audio_dist = {}
            self.combined_dist = {}
            return

        self.visual_dist = (
            self.df["visual_emotion"].value_counts(normalize=True) * 100
        ).to_dict() if "visual_emotion" in self.df.columns else {}

        self.audio_dist = (
            self.df["audio_emotion"].value_counts(normalize=True) * 100
        ).to_dict() if "audio_emotion" in self.df.columns else {}

        self.combined_dist = (
            self.df["combined_emotion"].value_counts(normalize=True) * 100
        ).to_dict() if "combined_emotion" in self.df.columns else {}

    @property
    def dominant_emotion(self) -> str:
        if not self.combined_dist:
            return "unknown"
        return max(self.combined_dist, key=self.combined_dist.get)

    @property
    def dominant_pct(self) -> float:
        if not self.combined_dist:
            return 0.0
        return self.combined_dist.get(self.dominant_emotion, 0.0)

    def get_pct(self, emotion: str, source: str = "combined") -> float:
        dist = {
            "visual": self.visual_dist,
            "audio": self.audio_dist,
            "combined": self.combined_dist,
        }.get(source, self.combined_dist)
        return dist.get(emotion, 0.0)

    def get_modality_agreement_rate(self) -> float:
        """Retorna a taxa de concordância entre visual e áudio (0–100%)."""
        if "visual_emotion" not in self.df.columns or "audio_emotion" not in self.df.columns:
            return 0.0
        agree = (self.df["visual_emotion"] == self.df["audio_emotion"]).sum()
        return (agree / len(self.df)) * 100 if len(self.df) > 0 else 0.0


# ============================================================
# CASOS DE USO CLÍNICOS
# ============================================================

class DepressionMonitor:
    """
    Monitora sinais vocais e expressivos compatíveis com Transtorno
    Depressivo Maior (CID-11: 6A70, DSM-5: 296.2x).

    Indicadores monitorados:
    - Humor deprimido: tristeza prolongada (> 50% do tempo)
    - Diminuição de energia: amplitude vocal reduzida
    - Retardo psicomotor: ritmo de fala lento
    - Anedonia: ausência de expressões positivas
    - Monotonia afetiva: baixa variação prosódica
    """

    def analyze(self, profile: EmotionProfile, audio_features: Optional[dict] = None) -> dict:
        sad_pct = profile.get_pct("sad")
        happy_pct = profile.get_pct("happy")
        energy = audio_features.get("energy_mean", 0.003) if audio_features else 0.003
        pitch_std = audio_features.get("pitch_std", 200) if audio_features else 200

        # Score de risco (0–10)
        score = 0.0

        # Tristeza prolongada
        if sad_pct > 70:
            score += 4.0
        elif sad_pct > 50:
            score += 2.5
        elif sad_pct > 30:
            score += 1.0

        # Baixa energia vocal
        if energy < 0.002:
            score += 2.5
        elif energy < 0.0025:
            score += 1.5
        elif energy < 0.003:
            score += 0.5

        # Monotonia prosódica (baixa variação de pitch)
        if pitch_std < 80:
            score += 2.0
        elif pitch_std < 150:
            score += 1.0

        # Ausência de afeto positivo
        if happy_pct < 5:
            score += 1.0

        score = min(score, 10.0)

        level = "normal"
        if score >= 8.5:
            level = "critical"
        elif score >= 7.0:
            level = "high"
        elif score >= 5.0:
            level = "moderate"
        elif score >= 3.0:
            level = "low"

        recommendations = []
        if level in ("low", "moderate"):
            recommendations.append("Monitoramento contínuo do estado afetivo recomendado")
            recommendations.append("Aplicar escala de avaliação (PHQ-9 ou BDI-II)")
        if level in ("high", "critical"):
            recommendations.append("Encaminhamento urgente a psiquiatra ou psicólogo clínico")
            recommendations.append("Avaliar risco de automutilação ou suicídio")
            recommendations.append("Considerar início de farmacoterapia antidepressiva")

        return {
            "condition": "Depressão",
            "cid_code": "CID-11: 6A70 | DSM-5: 296.2x",
            "risk_level": level,
            "risk_score": round(score, 2),
            "key_indicators": {
                "sad_percentage": round(sad_pct, 1),
                "happy_percentage": round(happy_pct, 1),
                "voice_energy": round(energy, 5),
                "pitch_variation": round(pitch_std, 1),
            },
            "recommendations": recommendations,
        }


class AnxietyDetector:
    """
    Detecta padrões vocais e expressivos compatíveis com Transtornos
    de Ansiedade (CID-11: 6B00–6B0Z, DSM-5: 300.xx).

    Indicadores:
    - Frequência de medo/alarme
    - Irregularidade vocal (ZCR elevado = voz trêmula)
    - Incongruência emocional audio/visual
    - Labilidade afetiva
    """

    def analyze(self, profile: EmotionProfile, audio_features: Optional[dict] = None,
                incongruence_score: float = 0.0) -> dict:
        fear_pct = profile.get_pct("fear")
        zcr = audio_features.get("zcr_mean", 0.05) if audio_features else 0.05
        pitch_std = audio_features.get("pitch_std", 200) if audio_features else 200

        score = 0.0

        # Frequência de medo
        if fear_pct > 50:
            score += 3.5
        elif fear_pct > 25:
            score += 2.0
        elif fear_pct > 10:
            score += 1.0

        # ZCR elevado (voz trêmula/tensa)
        if zcr > 0.12:
            score += 2.5
        elif zcr > 0.09:
            score += 1.5
        elif zcr > 0.07:
            score += 0.5

        # Variação excessiva de pitch (instabilidade vocal)
        if pitch_std > 800:
            score += 2.0
        elif pitch_std > 600:
            score += 1.0

        # Incongruência emocional
        if incongruence_score > 0.6:
            score += 2.0
        elif incongruence_score > 0.4:
            score += 1.0

        score = min(score, 10.0)

        level = "normal"
        if score >= 8.0:
            level = "critical"
        elif score >= 6.5:
            level = "high"
        elif score >= 4.5:
            level = "moderate"
        elif score >= 2.5:
            level = "low"

        recommendations = []
        if level == "low":
            recommendations.append("Aplicar escala GAD-7 para rastreamento de ansiedade")
        if level in ("moderate", "high"):
            recommendations.append("Avaliação clínica para Transtorno de Ansiedade Generalizada")
            recommendations.append("Considerar técnicas de manejo de ansiedade (TCC)")
        if level == "critical":
            recommendations.append("Avaliação emergencial - possível crise de pânico")
            recommendations.append("Intervenção medicamentosa de suporte (benzodiazepínico SOS)")

        return {
            "condition": "Ansiedade",
            "cid_code": "CID-11: 6B00 | DSM-5: 300.02",
            "risk_level": level,
            "risk_score": round(score, 2),
            "key_indicators": {
                "fear_percentage": round(fear_pct, 1),
                "voice_irregularity_zcr": round(zcr, 4),
                "pitch_instability": round(pitch_std, 1),
                "audiovisual_incongruence": round(incongruence_score, 3),
            },
            "recommendations": recommendations,
        }


class AgitationMonitor:
    """
    Detecta agitação psicomotora, compatível com episódios maníacos,
    transtorno explosivo intermitente ou outros estados de alta arousal.
    (CID-11: 6A60, DSM-5: 296.4x)
    """

    def analyze(self, profile: EmotionProfile, audio_features: Optional[dict] = None) -> dict:
        angry_pct = profile.get_pct("angry")
        energy = audio_features.get("energy_mean", 0.003) if audio_features else 0.003
        tempo = audio_features.get("tempo", 100) if audio_features else 100

        # Normaliza tempo
        if hasattr(tempo, "__iter__"):
            tempo = float(tempo[0]) if len(tempo) > 0 else 100.0

        score = 0.0

        if angry_pct > 50:
            score += 4.0
        elif angry_pct > 30:
            score += 2.5
        elif angry_pct > 15:
            score += 1.0

        if energy > 0.007:
            score += 2.5
        elif energy > 0.005:
            score += 1.5

        if tempo > 160:
            score += 2.0
        elif tempo > 140:
            score += 1.0

        score = min(score, 10.0)

        level = "normal"
        if score >= 8.0:
            level = "critical"
        elif score >= 6.0:
            level = "high"
        elif score >= 4.0:
            level = "moderate"
        elif score >= 2.5:
            level = "low"

        recommendations = []
        if level in ("moderate", "high"):
            recommendations.append("Avaliar possível episódio maníaco ou hipomaníaco")
            recommendations.append("Monitorar comportamento nas próximas horas")
        if level == "critical":
            recommendations.append("Intervenção imediata - risco de comportamento violento")
            recommendations.append("Considerar contenção verbal e medicamentosa")

        return {
            "condition": "Agitação Psicomotora",
            "cid_code": "CID-11: 6A60 | DSM-5: 296.4x",
            "risk_level": level,
            "risk_score": round(score, 2),
            "key_indicators": {
                "angry_percentage": round(angry_pct, 1),
                "voice_energy": round(energy, 5),
                "speech_tempo_bpm": round(float(tempo), 1),
            },
            "recommendations": recommendations,
        }


# ============================================================
# AVALIAÇÃO CLÍNICA CONSOLIDADA
# ============================================================

class ClinicalEvaluator:
    """
    Orquestra todas as avaliações clínicas e gera um perfil
    de risco completo para uso por profissionais de saúde.
    """

    def __init__(self):
        self.depression_monitor = DepressionMonitor()
        self.anxiety_detector = AnxietyDetector()
        self.agitation_monitor = AgitationMonitor()

    def evaluate(
        self,
        results_df: pd.DataFrame,
        audio_features: Optional[dict] = None,
        incongruence_score: float = 0.0,
        stability: str = "stable",
    ) -> dict:
        profile = EmotionProfile(results_df)

        depression = self.depression_monitor.analyze(profile, audio_features)
        anxiety = self.anxiety_detector.analyze(profile, audio_features, incongruence_score)
        agitation = self.agitation_monitor.analyze(profile, audio_features)

        # Determina nível de alerta geral
        levels_order = {"normal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
        all_levels = [depression["risk_level"], anxiety["risk_level"], agitation["risk_level"]]
        overall_level = max(all_levels, key=lambda x: levels_order.get(x, 0))

        # Alertas clínicos ativos
        active_alerts = []
        for assessment in [depression, anxiety, agitation]:
            if assessment["risk_level"] != "normal":
                active_alerts.append({
                    "condition": assessment["condition"],
                    "risk_level": assessment["risk_level"],
                    "score": assessment["risk_score"],
                    "recommendations": assessment["recommendations"],
                })

        if incongruence_score > 0.5:
            active_alerts.append({
                "condition": "Incongruência Emocional",
                "risk_level": "moderate",
                "score": incongruence_score * 10,
                "recommendations": [
                    "Emoção facial não corresponde ao tom de voz",
                    "Investigar regulação emocional prejudicada ou dissociação",
                    "Correlacionar com histórico clínico do paciente"
                ],
            })

        if stability == "unstable":
            active_alerts.append({
                "condition": "Labilidade Afetiva",
                "risk_level": "moderate",
                "score": 5.0,
                "recommendations": [
                    "Alta variação emocional em curto período",
                    "Avaliar transtorno de personalidade borderline (CID-11: 6D11)"
                ],
            })

        return {
            "overall_alert_level": overall_level,
            "emotion_profile": {
                "dominant_emotion": profile.dominant_emotion,
                "dominant_pct": round(profile.dominant_pct, 1),
                "combined_distribution": {k: round(v, 1) for k, v in profile.combined_dist.items()},
                "visual_distribution": {k: round(v, 1) for k, v in profile.visual_dist.items()},
                "audio_distribution": {k: round(v, 1) for k, v in profile.audio_dist.items()},
                "modality_agreement_pct": round(profile.get_modality_agreement_rate(), 1),
                "emotional_stability": stability,
                "incongruence_score": round(incongruence_score, 3),
            },
            "clinical_assessments": {
                "depression": depression,
                "anxiety": anxiety,
                "agitation": agitation,
            },
            "active_alerts": active_alerts,
            "evaluated_at": datetime.now().isoformat(),
        }

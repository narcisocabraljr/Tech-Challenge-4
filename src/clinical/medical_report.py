"""
Gerador de Relatório Médico Clínico - Tech Challenge 4
======================================================
Produz relatórios estruturados no formato de prontuário clínico,
compatível com documentação de sessões terapêuticas monitoradas.
"""

from datetime import datetime
from typing import Optional


RISK_LABELS = {
    "normal":   "✅ Normal",
    "low":      "🟡 Baixo",
    "moderate": "🟠 Moderado",
    "high":     "🔴 Alto",
    "critical": "🚨 Crítico",
}

RISK_COLORS = {
    "normal":   "#28a745",
    "low":      "#ffc107",
    "moderate": "#fd7e14",
    "high":     "#dc3545",
    "critical": "#6f0000",
}

EMOTION_PT = {
    "sad":      "Tristeza",
    "angry":    "Raiva",
    "fear":     "Medo",
    "happy":    "Alegria",
    "surprise": "Surpresa",
    "neutral":  "Neutro",
    "disgust":  "Nojo",
    "unknown":  "Desconhecida",
}


class MedicalReportGenerator:
    """
    Gera relatório clínico estruturado a partir da avaliação do ClinicalEvaluator.
    """

    def generate(
        self,
        clinical_evaluation: dict,
        patient_id: str = "N/D",
        patient_name: str = "Não Informado",
        session_date: Optional[str] = None,
        professional_name: str = "Não Informado",
        video_filename: str = "",
        total_frames: int = 0,
    ) -> str:
        """
        Gera relatório em formato Markdown clínico.

        Args:
            clinical_evaluation: Resultado do ClinicalEvaluator.evaluate()
            patient_id: Identificador do paciente
            patient_name: Nome do paciente
            session_date: Data da sessão (string ISO ou None para hoje)
            professional_name: Nome do profissional responsável
            video_filename: Nome do arquivo de vídeo analisado
            total_frames: Total de frames processados

        Returns:
            Relatório em formato Markdown
        """
        if session_date is None:
            session_date = datetime.now().strftime("%d/%m/%Y às %H:%M")

        ep = clinical_evaluation.get("emotion_profile", {})
        assessments = clinical_evaluation.get("clinical_assessments", {})
        alerts = clinical_evaluation.get("active_alerts", [])
        overall_level = clinical_evaluation.get("overall_alert_level", "normal")

        dominant_emotion_key = ep.get("dominant_emotion", "unknown")
        dominant_emotion_pt = EMOTION_PT.get(dominant_emotion_key, dominant_emotion_key.capitalize())
        dominant_pct = ep.get("dominant_pct", 0.0)
        agreement = ep.get("modality_agreement_pct", 0.0)
        stability = ep.get("emotional_stability", "N/D")
        incongruence = ep.get("incongruence_score", 0.0)

        lines = []

        # Cabeçalho
        lines += [
            "# 📋 Relatório de Análise Emocional Clínica",
            "",
            f"> **Nível de Alerta Geral:** {RISK_LABELS.get(overall_level, overall_level.upper())}",
            "",
            "---",
            "",
            "## 🏥 Informações da Sessão",
            "",
            f"| Campo | Valor |",
            f"|-------|-------|",
            f"| **ID do Paciente** | {patient_id} |",
            f"| **Nome** | {patient_name} |",
            f"| **Data / Hora** | {session_date} |",
            f"| **Profissional Responsável** | {professional_name} |",
            f"| **Arquivo Analisado** | `{video_filename}` |",
            f"| **Frames Processados** | {total_frames} |",
            "",
            "---",
            "",
            "## 📊 Perfil Emocional da Sessão",
            "",
            f"- **Emoção Dominante:** {dominant_emotion_pt} ({dominant_pct:.1f}%)",
            f"- **Estabilidade Emocional:** {stability}",
            f"- **Concordância Audio/Visual:** {agreement:.1f}%",
            f"- **Índice de Incongruência:** {incongruence:.3f}",
            "",
        ]

        # Distribuição combinada
        combined_dist = ep.get("combined_distribution", {})
        if combined_dist:
            lines += ["### Distribuição de Emoções (Análise Combinada)", ""]
            lines += ["| Emoção | % do Tempo |", "|--------|-----------|"]
            for emo, pct in sorted(combined_dist.items(), key=lambda x: -x[1]):
                emo_pt = EMOTION_PT.get(emo, emo)
                bar = "█" * int(pct / 5)
                lines.append(f"| {emo_pt} | {pct:.1f}% {bar} |")
            lines.append("")

        lines += ["---", ""]

        # Avaliações clínicas
        lines += ["## 🔬 Avaliações Clínicas", ""]

        for key, label in [("depression", "Depressão"), ("anxiety", "Ansiedade"), ("agitation", "Agitação")]:
            a = assessments.get(key, {})
            if not a:
                continue
            level_label = RISK_LABELS.get(a.get("risk_level", "normal"), "")
            score = a.get("risk_score", 0.0)
            cid = a.get("cid_code", "")

            lines += [
                f"### {label}",
                f"- **Classificação CID/DSM:** `{cid}`",
                f"- **Nível de Risco:** {level_label}",
                f"- **Score de Risco:** {score:.1f} / 10.0",
                "",
            ]

            indicators = a.get("key_indicators", {})
            if indicators:
                lines += ["**Indicadores Mensurados:**", ""]
                for k, v in indicators.items():
                    lines.append(f"- `{k}`: **{v}**")
                lines.append("")

            recs = a.get("recommendations", [])
            if recs:
                lines += ["**Recomendações:**", ""]
                for r in recs:
                    lines.append(f"- {r}")
                lines.append("")

        lines += ["---", ""]

        # Alertas ativos
        if alerts:
            lines += ["## ⚠️ Alertas Clínicos Ativos", ""]
            for alert in alerts:
                level_label = RISK_LABELS.get(alert.get("risk_level", "normal"), "")
                lines += [
                    f"### ⚠️ {alert.get('condition', '')} — {level_label}",
                    "",
                ]
                for r in alert.get("recommendations", []):
                    lines.append(f"- {r}")
                lines.append("")
            lines += ["---", ""]
        else:
            lines += [
                "## ✅ Sem Alertas Clínicos",
                "",
                "Nenhum indicador de risco significativo detectado nesta sessão.",
                "",
                "---",
                "",
            ]

        # Considerações técnicas e éticas
        lines += [
            "## ⚙️ Metodologia Técnica",
            "",
            "| Componente | Descrição |",
            "|------------|-----------|",
            "| **Detecção Visual** | YOLOv8 (detecção de pessoas) + DeepFace (expressões faciais) |",
            "| **Análise de Áudio** | Librosa — MFCCs, Pitch, Energia, Tempo, ZCR, Chroma (10 features) |",
            "| **Fusão Multimodal** | Fusão adaptativa com detecção de incongruência e pesos dinâmicos |",
            "| **Base de Dados** | RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) |",
            "",
            "---",
            "",
            "## ⚖️ Considerações Éticas e Limitações",
            "",
            "> **AVISO LEGAL:** Este relatório é gerado automaticamente por um sistema de IA e",
            "> tem caráter **auxiliar e complementar**, NÃO substituindo avaliação clínica",
            "> realizada por profissional de saúde habilitado (psicólogo ou psiquiatra).",
            "",
            "- Os resultados devem ser interpretados em conjunto com o histórico clínico completo",
            "- O sistema não realiza diagnóstico — apenas indica padrões que merecem atenção",
            "- A acurácia do sistema é influenciada pela qualidade do vídeo e condições de iluminação",
            "- Dados pessoais devem ser tratados conforme LGPD (Lei 13.709/2018) e CFM",
            "",
            "---",
            "",
            f"*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}*",
            f"*Sistema: Tech Challenge 4 - Análise Multimodal de Emoções*",
        ]

        return "\n".join(lines)

    def generate_summary_dict(self, clinical_evaluation: dict) -> dict:
        """
        Retorna um dicionário resumido com os principais indicadores clínicos.
        Útil para integração com dashboards e APIs.
        """
        ep = clinical_evaluation.get("emotion_profile", {})
        assessments = clinical_evaluation.get("clinical_assessments", {})
        alerts = clinical_evaluation.get("active_alerts", [])

        return {
            "overall_alert_level": clinical_evaluation.get("overall_alert_level", "normal"),
            "dominant_emotion": ep.get("dominant_emotion", "unknown"),
            "dominant_pct": ep.get("dominant_pct", 0.0),
            "modality_agreement_pct": ep.get("modality_agreement_pct", 0.0),
            "emotional_stability": ep.get("emotional_stability", "unknown"),
            "incongruence_score": ep.get("incongruence_score", 0.0),
            "depression_score": assessments.get("depression", {}).get("risk_score", 0.0),
            "depression_level": assessments.get("depression", {}).get("risk_level", "normal"),
            "anxiety_score": assessments.get("anxiety", {}).get("risk_score", 0.0),
            "anxiety_level": assessments.get("anxiety", {}).get("risk_level", "normal"),
            "agitation_score": assessments.get("agitation", {}).get("risk_score", 0.0),
            "agitation_level": assessments.get("agitation", {}).get("risk_level", "normal"),
            "active_alerts_count": len(alerts),
            "emotion_distribution": ep.get("combined_distribution", {}),
        }

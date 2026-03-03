"""
Dashboard Médico — Análise Multimodal de Emoções
=================================================
Interface web interativa construída com Streamlit para análise de emoções
em sessões de vídeo com contextualização clínica.

Uso:
    streamlit run app_streamlit.py

Funcionalidades:
    - Upload de vídeo para análise em tempo real
    - Visualização de resultados de CSVs pré-processados
    - Gráficos interativos de timeline emocional
    - Distribuição de emoções por modalidade (visual, áudio, combinada)
    - Avaliação clínica com scores de depressão, ansiedade e agitação
    - Relatório médico exportável em Markdown
    - Alertas clínicos interativos
"""

import os
import sys
import tempfile
import subprocess
import importlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Módulos internos ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Configuração da página ──────────────────────────────────
st.set_page_config(
    page_title="Sistema de Análise Emocional Clínica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ──────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    "#FFD700",
    "sad":      "#4169E1",
    "angry":    "#DC143C",
    "fear":     "#8B008B",
    "surprise": "#FF8C00",
    "neutral":  "#708090",
    "disgust":  "#556B2F",
    "unknown":  "#CCCCCC",
}

EMOTION_PT = {
    "happy":    "Alegria",
    "sad":      "Tristeza",
    "angry":    "Raiva",
    "fear":     "Medo",
    "surprise": "Surpresa",
    "neutral":  "Neutro",
    "disgust":  "Nojo",
    "unknown":  "Desconhecida",
}

RISK_COLORS_HEX = {
    "normal":   "#28a745",
    "low":      "#ffc107",
    "moderate": "#fd7e14",
    "high":     "#dc3545",
    "critical": "#6f0000",
}

RISK_LABEL_PT = {
    "normal":   "✅ Normal",
    "low":      "🟡 Baixo",
    "moderate": "🟠 Moderado",
    "high":     "🔴 Alto",
    "critical": "🚨 Crítico",
}


# ── Helpers ─────────────────────────────────────────────────

def _import_pipeline():
    """Importa módulos pesados de forma lazy para não travar a UI."""
    try:
        from audio_emotion_analyzer import AudioEmotionAnalyzer
        from src.multimodal_fusion import fuse_emotions_advanced, assess_audio_quality, assess_visual_quality
        from src.clinical import ClinicalEvaluator, MedicalReportGenerator
        from ultralytics import YOLO
        from deepface import DeepFace
        import cv2
        return AudioEmotionAnalyzer, fuse_emotions_advanced, assess_audio_quality, assess_visual_quality, ClinicalEvaluator, MedicalReportGenerator, YOLO, DeepFace, cv2
    except ImportError as e:
        return None


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=';')
    except Exception:
        try:
            return pd.read_csv(path, sep=',')
        except Exception:
            return pd.DataFrame()


def plot_emotion_timeline(df: pd.DataFrame, title: str = "Timeline Emocional") -> plt.Figure:
    all_emotions = list(EMOTION_COLORS.keys())
    df_plot = df.copy()
    df_plot["combined_emotion"] = df_plot["combined_emotion"].fillna("unknown")

    emotion_list = df_plot["combined_emotion"].unique().tolist()
    emotion_map = {e: i for i, e in enumerate(all_emotions)}
    df_plot["emotion_code"] = df_plot["combined_emotion"].map(
        lambda x: emotion_map.get(x, len(all_emotions))
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = [EMOTION_COLORS.get(e, "#CCCCCC") for e in df_plot["combined_emotion"]]

    scatter = ax.scatter(
        df_plot["frame"], df_plot["emotion_code"],
        c=colors, s=60, zorder=3
    )
    ax.plot(df_plot["frame"], df_plot["emotion_code"], alpha=0.3, color="gray", linewidth=1)

    ytick_indices = [emotion_map[e] for e in emotion_list if e in emotion_map]
    ytick_labels = [EMOTION_PT.get(e, e) for e in emotion_list if e in emotion_map]
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Emoção Detectada", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    return fig


def plot_distribution_bar(dist: dict, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_items = sorted(dist.items(), key=lambda x: -x[1])
    labels = [EMOTION_PT.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    bar_colors = [EMOTION_COLORS.get(k, "#CCCCCC") for k, _ in sorted_items]

    bars = ax.barh(labels, values, color=bar_colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=10)

    ax.set_xlabel("Percentual (%)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    return fig


def render_risk_badge(level: str, score: float, condition: str):
    color = RISK_COLORS_HEX.get(level, "#808080")
    label = RISK_LABEL_PT.get(level, level)
    st.markdown(
        f"""<div style="border-left: 5px solid {color}; padding: 10px 16px;
                        background-color: {color}18; border-radius: 6px; margin-bottom: 10px;">
            <strong>{condition}</strong> — {label} &nbsp;
            <span style="color:{color}; font-weight:bold;">({score:.1f}/10)</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=64)
    st.title("🧠 EmotionCare")
    st.caption("Sistema de Análise Emocional Clínica")
    st.divider()

    st.subheader("📋 Dados do Paciente")
    patient_id = st.text_input("ID do Paciente", value="PAC-001")
    patient_name = st.text_input("Nome", value="Paciente Anônimo")
    professional = st.text_input("Profissional Responsável", value="Dr(a). Não Informado")
    session_date = st.date_input("Data da Sessão")

    st.divider()
    st.subheader("⚙️ Modo de Uso")
    mode = st.radio(
        "Selecione o modo:",
        ["📂 Carregar Resultados Existentes", "🎬 Analisar Novo Vídeo"],
        index=0,
    )

    st.divider()
    st.caption("Tech Challenge 4 — FIAP")
    st.caption("Análise Multimodal de Emoções")


# ═══════════════════════════════════════════════════════════════
# CABEÇALHO PRINCIPAL
# ═══════════════════════════════════════════════════════════════

st.title("🏥 Sistema de Análise Emocional Clínica")
st.markdown(
    "Detecção automática de padrões emocionais em vídeos de sessões clínicas "
    "através de análise **multimodal** (expressão facial + características vocais)."
)
st.divider()


# ═══════════════════════════════════════════════════════════════
# MODO 1: CARREGAR RESULTADOS EXISTENTES
# ═══════════════════════════════════════════════════════════════

if mode == "📂 Carregar Resultados Existentes":

    csv_path = st.text_input(
        "Caminho do arquivo CSV",
        value="outputs/multimodal_emotions.csv",
        help="Arquivo gerado pelo emotion_pipeline.py"
    )

    if not os.path.exists(csv_path):
        st.warning(f"Arquivo não encontrado: `{csv_path}`. Execute o pipeline primeiro.")
        st.code("python emotion_pipeline.py", language="bash")
        st.stop()

    df = load_csv(csv_path)
    if df.empty:
        st.error("Arquivo CSV está vazio ou com formato inválido.")
        st.stop()

    # ── Filtro por vídeo ────────────────────────────────────
    all_videos = ["Todos os vídeos"] + sorted(df["video"].unique().tolist()) if "video" in df.columns else ["Todos os vídeos"]
    selected_video = st.selectbox("🎬 Filtrar por vídeo:", all_videos)

    if selected_video != "Todos os vídeos":
        df_view = df[df["video"] == selected_video].copy()
    else:
        df_view = df.copy()

    # ── Métricas de cabeçalho ────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    dominant = df_view["combined_emotion"].mode()[0] if not df_view.empty else "N/D"
    dominant_pct = (df_view["combined_emotion"] == dominant).mean() * 100

    incongruence_avg = df_view["incongruence_score"].mean() if "incongruence_score" in df_view.columns else 0.0

    agr_rate = 0.0
    if "visual_emotion" in df_view.columns and "audio_emotion" in df_view.columns:
        agr_rate = (df_view["visual_emotion"] == df_view["audio_emotion"]).mean() * 100

    col1.metric("📊 Frames Analisados", len(df_view))
    col2.metric("🎭 Emoção Dominante", f"{EMOTION_PT.get(dominant, dominant)} ({dominant_pct:.0f}%)")
    col3.metric("🤝 Concordância A/V", f"{agr_rate:.1f}%")
    col4.metric("⚡ Incongruência Média", f"{incongruence_avg:.3f}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Timeline Emocional",
        "📊 Distribuição de Emoções",
        "🏥 Avaliação Clínica",
        "📄 Relatório Médico",
    ])

    # Tab 1 – Timeline
    with tab1:
        st.subheader("Timeline Emocional Combinada")
        title = f"Timeline — {selected_video}" if selected_video != "Todos os vídeos" else "Timeline — Todos os vídeos"
        try:
            fig = plot_emotion_timeline(df_view, title)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Erro ao gerar timeline: {e}")

        if "confidence" in df_view.columns:
            st.subheader("Distribuição de Confiança da Fusão Multimodal")
            conf_counts = df_view["confidence"].value_counts()
            conf_pct = (conf_counts / conf_counts.sum() * 100).round(1)

            conf_labels = {
                "high_confidence": "✅ Alta Confiança (concordância A/V)",
                "audio_priority":  "🎙️ Prioridade Áudio (rosto neutro)",
                "visual_priority": "👁️ Prioridade Visual (voz neutra ou conflito)",
                "audio_only":      "🎵 Apenas Áudio",
                "visual_only":     "📷 Apenas Visual",
                "no_data":         "❌ Sem Dados",
            }

            for conf_type, pct in sorted(conf_pct.items(), key=lambda x: -x[1]):
                label = conf_labels.get(conf_type, conf_type)
                st.progress(int(pct), text=f"{label}: **{pct:.1f}%**")

    # Tab 2 – Distribuição
    with tab2:
        colA, colB, colC = st.columns(3)

        with colA:
            if "visual_emotion" in df_view.columns:
                dist_v = (df_view["visual_emotion"].value_counts(normalize=True) * 100).to_dict()
                fig = plot_distribution_bar(dist_v, "👁️ Expressão Facial (Visual)")
                st.pyplot(fig)
                plt.close(fig)

        with colB:
            if "audio_emotion" in df_view.columns:
                dist_a = (df_view["audio_emotion"].value_counts(normalize=True) * 100).to_dict()
                fig = plot_distribution_bar(dist_a, "🎙️ Análise Vocal (Áudio)")
                st.pyplot(fig)
                plt.close(fig)

        with colC:
            if "combined_emotion" in df_view.columns:
                dist_c = (df_view["combined_emotion"].value_counts(normalize=True) * 100).to_dict()
                fig = plot_distribution_bar(dist_c, "🔀 Fusão Multimodal (Combinada)")
                st.pyplot(fig)
                plt.close(fig)

        # Tabela comparativa
        st.subheader("📋 Tabela Comparativa por Emoção")
        all_emos = set()
        for col in ["visual_emotion", "audio_emotion", "combined_emotion"]:
            if col in df_view.columns:
                all_emos.update(df_view[col].dropna().unique())

        comp_rows = []
        for emo in sorted(all_emos):
            row = {"Emoção": EMOTION_PT.get(emo, emo)}
            for col, label in [("visual_emotion", "Visual (%)"), ("audio_emotion", "Áudio (%)"), ("combined_emotion", "Combinada (%)")]:
                if col in df_view.columns:
                    pct = (df_view[col] == emo).mean() * 100
                    row[label] = f"{pct:.1f}%"
            comp_rows.append(row)

        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    # Tab 3 – Avaliação Clínica
    with tab3:
        try:
            from src.clinical import ClinicalEvaluator, MedicalReportGenerator

            stability_col = "N/D"
            audio_summary_path = csv_path.replace("multimodal_emotions", "audio_analysis_summary")
            if os.path.exists(audio_summary_path):
                df_audio = load_csv(audio_summary_path)
                if not df_audio.empty and "emotional_stability" in df_audio.columns:
                    stability_col = df_audio["emotional_stability"].mode()[0]

            evaluator = ClinicalEvaluator()
            clinical_eval = evaluator.evaluate(
                results_df=df_view,
                incongruence_score=float(incongruence_avg),
                stability=str(stability_col),
            )

            ep = clinical_eval["emotion_profile"]
            assessments = clinical_eval["clinical_assessments"]
            alerts = clinical_eval["active_alerts"]
            overall = clinical_eval["overall_alert_level"]

            # Alerta geral
            overall_color = RISK_COLORS_HEX.get(overall, "#808080")
            overall_label = RISK_LABEL_PT.get(overall, overall)
            st.markdown(
                f"""<div style="padding: 16px; border-radius: 10px;
                               background-color: {overall_color}22;
                               border: 2px solid {overall_color}; margin-bottom: 20px;">
                    <h3 style="margin:0;">🏥 Nível de Alerta Geral: {overall_label}</h3>
                    <p style="margin:4px 0 0 0; color: #555;">
                        Emoção dominante: <strong>{EMOTION_PT.get(ep['dominant_emotion'], ep['dominant_emotion'])}</strong>
                        ({ep['dominant_pct']:.1f}%) &nbsp;|&nbsp;
                        Concordância A/V: <strong>{ep['modality_agreement_pct']:.1f}%</strong>
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Scores clínicos
            st.subheader("📊 Scores de Risco Clínico")
            col1, col2, col3 = st.columns(3)

            dep = assessments["depression"]
            anx = assessments["anxiety"]
            agi = assessments["agitation"]

            col1.metric(
                "Depressão", f"{dep['risk_score']:.1f}/10",
                delta=RISK_LABEL_PT.get(dep["risk_level"], ""),
                delta_color="off"
            )
            col2.metric(
                "Ansiedade", f"{anx['risk_score']:.1f}/10",
                delta=RISK_LABEL_PT.get(anx["risk_level"], ""),
                delta_color="off"
            )
            col3.metric(
                "Agitação", f"{agi['risk_score']:.1f}/10",
                delta=RISK_LABEL_PT.get(agi["risk_level"], ""),
                delta_color="off"
            )

            st.subheader("🔬 Detalhes das Avaliações")
            for key, label in [("depression", "Depressão"), ("anxiety", "Ansiedade"), ("agitation", "Agitação Psicomotora")]:
                a = assessments[key]
                with st.expander(f"{label} — {RISK_LABEL_PT.get(a['risk_level'], a['risk_level'])} | Score: {a['risk_score']:.1f}/10"):
                    st.caption(f"**Código CID/DSM:** {a.get('cid_code', 'N/D')}")
                    st.markdown("**Indicadores Mensurados:**")
                    for k, v in a.get("key_indicators", {}).items():
                        st.markdown(f"- `{k}`: **{v}**")
                    if a.get("recommendations"):
                        st.markdown("**Recomendações:**")
                        for r in a["recommendations"]:
                            st.markdown(f"- {r}")

            # Alertas ativos
            if alerts:
                st.subheader("⚠️ Alertas Clínicos Ativos")
                for alert in alerts:
                    level = alert.get("risk_level", "normal")
                    if level in ("critical", "high"):
                        alert_fn = st.error
                    elif level == "moderate":
                        alert_fn = st.warning
                    else:
                        alert_fn = st.info

                    with alert_fn(f"**{alert['condition']}** — {RISK_LABEL_PT.get(level, level)}"):
                        for r in alert.get("recommendations", []):
                            st.markdown(f"- {r}")
            else:
                st.success("✅ Nenhum alerta clínico ativo — perfil dentro dos parâmetros normais.")

            # Ética
            st.divider()
            st.info(
                "⚖️ **Aviso Legal:** Este sistema é auxiliar e NÃO substitui avaliação clínica "
                "por profissional habilitado (psicólogo ou psiquiatra). Os dados devem ser "
                "tratados conforme LGPD (Lei 13.709/2018).",
            )

            # Armazena para a aba de relatório
            st.session_state["clinical_eval"] = clinical_eval
            st.session_state["df_view"] = df_view

        except ImportError:
            st.error("Módulo clínico não encontrado. Verifique a estrutura src/clinical/.")
        except Exception as e:
            st.error(f"Erro na avaliação clínica: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 4 – Relatório
    with tab4:
        st.subheader("📄 Relatório Clínico Estruturado")

        if "clinical_eval" not in st.session_state:
            st.info("Acesse a aba **🏥 Avaliação Clínica** primeiro para gerar o relatório.")
        else:
            try:
                reporter = MedicalReportGenerator()
                report_md = reporter.generate(
                    clinical_evaluation=st.session_state["clinical_eval"],
                    patient_id=patient_id,
                    patient_name=patient_name,
                    session_date=session_date.strftime("%d/%m/%Y"),
                    professional_name=professional,
                    video_filename=selected_video if selected_video != "Todos os vídeos" else "Múltiplos vídeos",
                    total_frames=len(st.session_state["df_view"]),
                )
                st.markdown(report_md)
                st.divider()

                st.download_button(
                    label="📥 Baixar Relatório (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=f"relatorio_clinico_{patient_id}_{session_date}.md",
                    mime="text/markdown",
                )

                csv_export = st.session_state["df_view"].to_csv(index=False, sep=';').encode("utf-8")
                st.download_button(
                    label="📥 Baixar Dados da Sessão (.csv)",
                    data=csv_export,
                    file_name=f"dados_sessao_{patient_id}_{session_date}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {e}")


# ═══════════════════════════════════════════════════════════════
# MODO 2: ANALISAR NOVO VÍDEO
# ═══════════════════════════════════════════════════════════════

elif mode == "🎬 Analisar Novo Vídeo":

    st.subheader("🎬 Analisar Novo Vídeo")
    st.info(
        "**Nota:** A análise completa pode levar entre 30 segundos e 2 minutos "
        "dependendo do tamanho do vídeo e do hardware disponível."
    )

    uploaded = st.file_uploader(
        "Faça upload do vídeo da sessão",
        type=["mp4", "avi", "mov", "mkv"],
        help="Formatos suportados: MP4, AVI, MOV, MKV"
    )

    if uploaded:
        st.video(uploaded)

        if st.button("🔍 Iniciar Análise Multimodal", type="primary"):
            # Escreve arquivo temporário
            suffix = os.path.splitext(uploaded.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                modules = _import_pipeline()
                if modules is None:
                    st.error("Dependências não encontradas. Instale os requirements: `pip install -r requirements.txt`")
                    st.stop()

                (AudioEmotionAnalyzer, fuse_emotions_advanced, assess_audio_quality,
                 assess_visual_quality, ClinicalEvaluator, MedicalReportGenerator,
                 YOLO, DeepFace, cv2) = modules

                progress_bar = st.progress(0, text="Inicializando modelos...")

                @st.cache_resource
                def get_yolo():
                    return YOLO("yolov8n.pt")

                yolo_model = get_yolo()
                audio_analyzer = AudioEmotionAnalyzer()

                # Análise de áudio
                progress_bar.progress(15, text="Extraindo áudio...")
                audio_path = audio_analyzer.extract_audio_from_video(tmp_path)
                audio_result = None
                if audio_path:
                    progress_bar.progress(30, text="Analisando características vocais...")
                    audio_result = audio_analyzer.process_audio(audio_path)
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

                audio_features = audio_result.get("features") if audio_result else None
                audio_qual = assess_audio_quality(audio_features)
                audio_emotion = audio_result["emotion"] if audio_result else None

                # Análise visual
                progress_bar.progress(45, text="Analisando expressões faciais...")
                cap = cv2.VideoCapture(tmp_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                FRAME_SKIP = 10
                results = []
                frame_id = 0
                processed = 0

                for _ in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_id += 1
                    if frame_id % FRAME_SKIP != 0:
                        continue

                    detections = yolo_model(frame, verbose=False)[0]
                    visual_emotion = None

                    for box in detections.boxes:
                        if box.conf < 0.5:
                            continue
                        cls = int(box.cls[0])
                        if yolo_model.names[cls] != "person":
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                        try:
                            r = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
                            visual_emotion = r[0]["dominant_emotion"]
                        except Exception:
                            pass
                        break

                    visual_qual = assess_visual_quality(0.7, visual_emotion is not None)
                    combined, confidence, incongruence, vw, aw = fuse_emotions_advanced(
                        visual_emotion, audio_emotion, visual_qual, audio_qual
                    )

                    results.append({
                        "frame": frame_id,
                        "visual_emotion": visual_emotion,
                        "audio_emotion": audio_emotion,
                        "combined_emotion": combined,
                        "confidence": confidence,
                        "incongruence_score": round(incongruence, 3),
                        "visual_weight": round(vw, 3),
                        "audio_weight": round(aw, 3),
                    })
                    processed += 1
                    prog = 45 + int((processed / max(frame_count // FRAME_SKIP, 1)) * 45)
                    progress_bar.progress(min(prog, 90), text=f"Processando frames... ({frame_id}/{frame_count})")

                cap.release()
                progress_bar.progress(95, text="Gerando avaliação clínica...")

                df_res = pd.DataFrame(results)
                incongruence_avg = df_res["incongruence_score"].mean() if not df_res.empty else 0.0

                evaluator = ClinicalEvaluator()
                clinical_eval = evaluator.evaluate(
                    results_df=df_res,
                    audio_features=audio_features,
                    incongruence_score=float(incongruence_avg),
                )

                progress_bar.progress(100, text="✅ Análise concluída!")
                st.success(f"✅ {len(results)} frames processados com sucesso!")

                # ── Exibe resultados ────────────────────────────────
                ep = clinical_eval["emotion_profile"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Emoção Dominante", EMOTION_PT.get(ep["dominant_emotion"], ep["dominant_emotion"]))
                col2.metric("Concordância A/V", f"{ep['modality_agreement_pct']:.1f}%")
                col3.metric("Alerta Geral", RISK_LABEL_PT.get(clinical_eval["overall_alert_level"], ""))

                if not df_res.empty:
                    fig = plot_emotion_timeline(df_res, f"Timeline — {uploaded.name}")
                    st.pyplot(fig)
                    plt.close(fig)

                # Relatório
                reporter = MedicalReportGenerator()
                report_md = reporter.generate(
                    clinical_evaluation=clinical_eval,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    session_date=session_date.strftime("%d/%m/%Y"),
                    professional_name=professional,
                    video_filename=uploaded.name,
                    total_frames=len(df_res),
                )

                with st.expander("📄 Ver Relatório Clínico Completo"):
                    st.markdown(report_md)

                st.download_button(
                    "📥 Baixar Relatório (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=f"relatorio_{patient_id}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Erro durante a análise: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

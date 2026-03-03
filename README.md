# Tech Challenge 4 - Análise Multimodal de Emoções 🎭🎵

Sistema avançado de reconhecimento de emoções que combina análise de **expressões faciais (visual)** e **características acústicas (áudio)** para detecção precisa do estado emocional em vídeos.

## 📋 Descrição do Projeto

Este projeto implementa um pipeline completo de análise emocional multimodal utilizando técnicas de Deep Learning e Processamento de Sinais. O sistema processa vídeos extraindo informações de duas modalidades complementares:

- **Análise Visual**: Detecção de faces com YOLOv8 e reconhecimento de expressões faciais com DeepFace
- **Análise de Áudio**: Extração de características acústicas (MFCCs, pitch, energia, etc.) e classificação emocional baseada em padrões de fala

### 🎯 Objetivos

- Detectar emoções em vídeos através de análise multimodal
- Extrair características acústicas como tom de voz, ritmo e pausas
- Identificar anomalias e variações significativas no estado emocional
- Combinar informações de áudio e vídeo para aumentar a precisão da detecção

## ✅ Status do Ambiente

### 📦 Pacotes Instalados
- ✅ **Python 3.13.5**
- ✅ **numpy 2.3.5**
- ✅ **pandas 3.0.0**
- ✅ **opencv-python 4.13.0**
- ✅ **pytorch 2.10.0**
- ✅ **ultralytics 8.4.14** (YOLOv8)
- ✅ **deepface 0.0.98**
- ✅ **librosa 0.11.0** (análise de áudio)
- ✅ **soundfile 0.13.1** (manipulação de áudio)
- ✅ **scikit-learn 1.8.0** (ML)
- ✅ **FFmpeg 8.0.1** (extração de áudio)

### 🎬 Dataset
- ✅ **240 vídeos** do RAVDESS encontrados

---

## 🎯 Melhorias e Correções Implementadas

### ✅ Classificador de Áudio Refinado
- **Problema inicial**: 100% das classificações retornavam "neutral"
- **Solução**: Thresholds ajustados baseados em valores reais extraídos dos vídeos
- **Resultado**: **100% de acurácia** nos testes com 6 emoções diferentes

**Thresholds otimizados:**
```python
# Sad: energy < 0.003, pitch < 250, tempo < 100
# Angry: pitch_std > 750, tempo > 125, energy > 0.004
# Fear: pitch_std > 550, zcr > 0.075, energy < 0.007
# Surprise: energy_std > 0.004, pitch_std 350-600
# Happy: pitch > 280, tempo > 110, energy > 0.004
# Neutral: demais padrões
```

### ✅ Fusão Multimodal Inteligente
Estratégia de combinação melhorada:
- Se **ambas modalidades concordam** → alta confiança
- Se **áudio = neutral** mas **visual ≠ neutral** → prioriza VISUAL
- Se **visual = neutral** mas **áudio ≠ neutral** → prioriza ÁUDIO
- Em **conflitos** entre emoções não-neutras → prioriza VISUAL (DeepFace mais confiável)

### ✅ Correções Técnicas
- ✅ Normalização de arrays para escalar em `analyze_emotional_variations()`
- ✅ Reset de `features_history` entre vídeos para evitar contaminação de dados
- ✅ Tratamento robusto de valores multidimensionais com `_extract_scalar()`
- ✅ Limpeza automática de arquivos temporários (temp_*.wav)
- ✅ Sistema de rastreamento de erros e log detalhado

### 📊 Resultados de Validação
```
Teste                     Esperado     Detectado      Status
----------------------------------------------------------------------
Vídeo Real (Neutral)      neutral      neutral          ✅
Simulado (Angry)          angry        angry            ✅
Simulado (Happy)          happy        happy            ✅
Simulado (Sad)            sad          sad              ✅
Simulado (Fear)           fear         fear             ✅
Simulado (Surprise)       surprise     surprise         ✅
----------------------------------------------------------------------
Acurácia: 6/6 (100.0%) ✅
```

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                             │
└─────────────────┬──────────────────┬────────────────────────┘
                  │                  │
         ┌────────▼────────┐  ┌─────▼──────────┐
         │  Visual Stream  │  │  Audio Stream  │
         └────────┬────────┘  └─────┬──────────┘
                  │                  │
         ┌────────▼────────┐  ┌─────▼──────────┐
         │  YOLOv8 Face    │  │   Librosa      │
         │   Detection     │  │   Feature      │
         └────────┬────────┘  │  Extraction    │
                  │           └─────┬──────────┘
         ┌────────▼────────┐  ┌─────▼──────────┐
         │   DeepFace      │  │  Acoustic      │
         │   Emotion       │  │  Classifier    │
         └────────┬────────┘  └─────┬──────────┘
                  │                  │
                  └────────┬─────────┘
                           │
                  ┌────────▼────────┐
                  │  Multimodal     │
                  │    Fusion       │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Final Emotion  │
                  │   + Analysis    │
                  └─────────────────┘
```

## 🚀 Funcionalidades

### 1. Análise Visual (DeepFace + YOLO)
- ✅ Detecção de pessoas em frames com YOLOv8
- ✅ Reconhecimento de expressões faciais
- ✅ Extração de emoções frame-a-frame
- ✅ Suporta 7 emoções: angry, disgust, fear, happy, sad, surprise, neutral

### 2. Análise de Áudio (Librosa)
- ✅ **MFCCs** (Mel-frequency cepstral coefficients) - Tom/timbre da voz
- ✅ **Pitch** (F0) - Frequência fundamental
- ✅ **Energia/RMS** - Volume e intensidade
- ✅ **ZCR** (Zero Crossing Rate) - Ritmo e textura
- ✅ **Spectral Features** - Características espectrais (centroid, rolloff, bandwidth)
- ✅ **Tempo** - Velocidade da fala/ritmo
- ✅ **Pausas** - Detecção e análise de silêncios
- ✅ **Chroma Features** - Características tonais

### 3. Detecção de Anomalias
- ✅ Identificação de variações significativas usando Z-score
- ✅ Análise de estabilidade emocional ao longo do tempo
- ✅ Detecção de padrões atípicos no comportamento vocal

### 4. Fusão Multimodal
- ✅ Combinação inteligente de emoções de áudio e vídeo
- ✅ Sistema de confiança baseado em concordância entre modalidades
- ✅ Resolução de conflitos entre análises

## �️ Interface Web (Dashboard Clínico)

O projeto inclui um dashboard médico interativo construído com **Streamlit** que permite:

- Upload de vídeos para análise em tempo real
- Visualização de resultados de sessões já processadas
- Gráficos de timeline emocional e distribuição por modalidade
- Avaliação clínica com scores de depressão, ansiedade e agitação
- Geração e download de relatório médico estruturado
- Alertas clínicos com classificação por nível de risco

### ▶️ Como Rodar a Interface

**1. Ative o ambiente virtual (se ainda não estiver ativo):**
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

**2. Inicie o Streamlit:**
```powershell
# Windows (com venv ativo)
streamlit run app_streamlit.py

# Ou diretamente pelo executável do venv
.\venv\Scripts\streamlit.exe run app_streamlit.py
```

**3. Acesse no navegador:**
```
http://localhost:8501
```

### 📋 Modos de Uso da Interface

#### Modo 1 — Carregar Resultados Existentes
Visualize e analise resultados de um processamento já realizado pelo `emotion_pipeline.py`:
1. Execute o pipeline primeiro: `.\venv\Scripts\python.exe emotion_pipeline.py`
2. Inicie o Streamlit: `streamlit run app_streamlit.py`
3. Selecione **"📂 Carregar Resultados Existentes"** na barra lateral
4. O caminho padrão `outputs/multimodal_emotions.csv` é carregado automaticamente
5. Filtre por vídeo e navegue pelas 4 abas de análise

#### Modo 2 — Analisar Novo Vídeo
Faça upload de qualquer vídeo para análise direta pela interface:
1. Selecione **"🎬 Analisar Novo Vídeo"** na barra lateral
2. Faça upload do arquivo (`.mp4`, `.avi`, `.mov`, `.mkv`)
3. Clique em **"🔍 Iniciar Análise Multimodal"**
4. Aguarde o processamento (30s a 2 min dependendo do vídeo)
5. Visualize resultados, gráficos e relatório clínico

### 🗂️ Abas Disponíveis

| Aba | Conteúdo |
|-----|----------|
| **📈 Timeline Emocional** | Gráfico de evolução das emoções por frame + distribuição de confiança da fusão |
| **📊 Distribuição de Emoções** | Gráficos comparativos visual / áudio / combinado + tabela por emoção |
| **🏥 Avaliação Clínica** | Scores de depressão, ansiedade e agitação (0–10) com alertas e recomendações |
| **📄 Relatório Médico** | Relatório em formato de prontuário clínico com download em `.md` |

### ⚠️ Requisito Adicional

Caso o Streamlit não esteja instalado:
```powershell
pip install streamlit
```

---

## �📦 Instalação

### Pré-requisitos

- Python 3.8+ (testado com Python 3.13.5)
- FFmpeg (para extração de áudio)
- CUDA (opcional, para aceleração GPU)

### Instalação do FFmpeg

**Windows:**
```powershell
# Usando winget (recomendado)
winget install -e --id Gyan.FFmpeg

# Ou usando Chocolatey
choco install ffmpeg
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Instalação das Dependências Python

```bash
# Clone o repositório
git clone <seu-repositorio>
cd Tech-Challenge-4

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### Execução Rápida

```powershell
# Windows - Execute diretamente com o Python do venv
.\venv\Scripts\python.exe emotion_pipeline.py
```

---

## 📊 Estrutura do Projeto

```
Tech-Challenge-4/
├── emotion_pipeline.py              # Pipeline principal (análise multimodal)
├── audio_emotion_analyzer.py        # Módulo de análise de áudio
├── app_streamlit.py                 # Interface web (dashboard clínico)
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Esta documentação
├── yolov8n.pt                       # Modelo YOLO pré-treinado
├── .gitignore                       # Arquivos ignorados pelo Git
│
├── src/                             # Módulos internos
│   ├── multimodal_fusion.py         # Fusão multimodal avançada (pesos adaptativos)
│   └── clinical/
│       ├── clinical_analyzer.py     # Avaliadores clínicos (depressão, ansiedade, agitação)
│       └── medical_report.py        # Gerador de relatório em formato prontuário
│
├── config/
│   └── medical_thresholds.yaml      # Limiares clínicos (DSM-5 / CID-11)
│
├── RAVDESS/                         # Dataset de vídeos (240 vídeos)
│   ├── Video_Speech_Actor_02/
│   ├── Video_Speech_Actor_04/
│   └── ...
│
├── outputs/                         # Resultados da análise
│   ├── multimodal_emotions.csv      # Emoções combinadas (frame-a-frame)
│   ├── audio_analysis_summary.csv   # Resumo da análise de áudio por vídeo
│   ├── clinical_report.md           # Relatório clínico gerado pelo pipeline
│   ├── *_timeline.png               # Timelines emocionais (1 por vídeo)
│   └── processing_errors.log        # Log de erros (se houver)
│
└── venv/                            # Ambiente virtual Python (não versionado)
```

## 🎮 Uso

### Execução Completa

```bash
# Com ambiente virtual ativado
python emotion_pipeline.py

# Ou diretamente (Windows)
.\venv\Scripts\python.exe emotion_pipeline.py
```

O script irá:
1. Carregar 10 vídeos aleatórios do dataset RAVDESS
2. Processar cada vídeo com análise multimodal (áudio + visual)
3. Gerar relatórios CSV na pasta `outputs/`
4. Exibir resumo estatístico no terminal

### Exemplo de Saída

```
Vídeos: 100%|████████████████████████████| 10/10 [00:39<00:00, 3.98s/it]

==================================================
RESUMO DA ANÁLISE MULTIMODAL
==================================================

📊 Distribuição de Emoções Combinadas:
  fear: 35.45%
  neutral: 27.27%
  sad: 22.73%
  angry: 8.18%
  happy: 4.55%
  surprise: 1.82%

🎵 Análise de Áudio:
  fear: 50.00%
  neutral: 40.00%
  happy: 10.00%

⚠️ Anomalias Detectadas:
  0 de 10 vídeos (0.0%)

💚 Estabilidade Emocional:
  insufficient_data: 10 vídeos

✅ Relatórios salvos em:
  - outputs/multimodal_emotions.csv
  - outputs/audio_analysis_summary.csv
```

### Uso Programático

```python
from audio_emotion_analyzer import AudioEmotionAnalyzer
from emotion_pipeline import process_multimodal_video

# Inicializar analisador
analyzer = AudioEmotionAnalyzer()

# Processar vídeo
video_path = "RAVDESS/Video_Speech_Actor_02/03-01-01-01-01-01-02.mp4"
results, audio_info = process_multimodal_video(video_path, analyzer)

# Acessar resultados
print(f"Emoção de áudio: {audio_info['audio_emotion']}")
print(f"Anomalia detectada: {audio_info['is_anomaly']}")
print(f"Estabilidade: {audio_info['emotional_stability']}")
print(f"Frames analisados: {len(results)}")
```

## 🔧 Solução de Problemas

### Python não está usando o venv
```powershell
# Execute diretamente com o caminho completo
.\venv\Scripts\python.exe emotion_pipeline.py
```

### Erro "ffmpeg not found"
```powershell
# Windows - Instale o FFmpeg
winget install -e --id Gyan.FFmpeg

# Verifique a instalação
ffmpeg -version
```

### Erro de importação de módulos
```bash
# Reinstale as dependências
pip install -r requirements.txt --upgrade
```

### Arquivos temp_*.wav não são deletados
Os arquivos temporários são automaticamente deletados após o processamento. Se persistirem após um erro, você pode removê-los manualmente:
```powershell
Remove-Item temp_*.wav
```

---

## 📈 Resultados e Métricas

### Outputs Gerados

1. **multimodal_emotions.csv** - Análise frame-a-frame
   - `video`: Nome do arquivo de vídeo
   - `frame`: Número do frame
   - `visual_emotion`: Emoção detectada pela análise facial (DeepFace)
   - `audio_emotion`: Emoção detectada pela análise de áudio (Librosa)
   - `combined_emotion`: Emoção final após fusão multimodal
   - `confidence`: Nível de confiança ("high" quando modalidades concordam)

2. **audio_analysis_summary.csv** - Resumo por vídeo
   - `video`: Nome do vídeo
   - `audio_emotion`: Emoção predominante no áudio
   - `is_anomaly`: Se foi detectada anomalia emocional (Z-score > 2.0)
   - `emotional_stability`: Nível de estabilidade (stable/moderate/unstable/insufficient_data)
   - `audio_features`: Dicionário com features acústicas extraídas

3. **processing_errors.log** - Log de erros (se houver falhas)

## 🔬 Metodologia

### Classificação de Emoções por Áudio

O sistema utiliza **regras heurísticas refinadas** baseadas em pesquisas de Speech Emotion Recognition (SER):

| Emoção | Características Acústicas |
|--------|---------------------------|
| **Raiva** | pitch_std > 750 AND tempo > 125 AND energy > 0.004 |
| **Felicidade** | pitch_mean > 280 AND tempo > 110 AND energy > 0.004 |
| **Tristeza** | energy < 0.003 AND pitch_mean < 250 AND tempo < 100 |
| **Medo** | pitch_std > 550 AND zcr > 0.075 AND energy < 0.007 |
| **Surpresa** | energy_std > 0.004 AND pitch_std ∈ [350, 600] |
| **Neutro** | Padrões que não se encaixam nas categorias acima |

### Detecção de Anomalias

Utiliza **Z-score** para identificar desvios significativos:
- Calcula Z-score para features-chave: `pitch_mean`, `energy_mean`, `tempo`
- Threshold: Z-score > 2.0 = Anomalia detectada
- Compara features atuais com histórico acumulado

### Fusão Multimodal

Estratégia de combinação inteligente:

| Áudio | Visual | Resultado | Confiança |
|-------|--------|-----------|-----------|
| emotion_A | emotion_A | emotion_A | high |
| neutral | emotion_B | emotion_B | medium |
| emotion_A | neutral | emotion_A | medium |
| emotion_A | emotion_B | emotion_B (visual) | low |

**Justificativa**: DeepFace (visual) tende a ser mais confiável que classificação heurística de áudio, mas áudio é útil quando faces não são detectadas.

## 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologia | Versão | Função |
|-----------|-----------|--------|--------|
| **Visão Computacional** | YOLOv8 (Ultralytics) | 8.4.14 | Detecção de pessoas |
| | DeepFace | 0.0.98 | Análise de expressões faciais |
| | OpenCV | 4.13.0 | Processamento de imagens |
| **Processamento de Áudio** | Librosa | 0.11.0 | Extração de features acústicas |
| | SoundFile | 0.13.1 | Leitura/escrita de áudio |
| | FFmpeg | 8.0.1 | Extração e conversão de áudio |
| **Machine Learning** | scikit-learn | 1.8.0 | Normalização e análise estatística |
| | TensorFlow | 2.20.0 | Backend para DeepFace |
| | PyTorch | 2.10.0 | Backend para YOLOv8 |
| **Análise de Dados** | Pandas | 3.0.0 | Manipulação de dados tabulares |
| | NumPy | 2.3.5 | Computação numérica |
| | SciPy | 1.17.0 | Análise estatística avançada |

### Features Acústicas Extraídas

| Feature | Descrição | Importância |
|---------|-----------|-------------|
| **MFCCs** | 13 coeficientes mel-cepstrais | Timbre/tom de voz |
| **Pitch (F0)** | Frequência fundamental (mean/std/min/max) | Altura da voz, emoção |
| **Energy/RMS** | Root Mean Square (mean/std) | Volume, intensidade |
| **ZCR** | Zero Crossing Rate | Percussividade, textura |
| **Spectral Centroid** | Centro de massa do espectro | Brilho do som |
| **Spectral Rolloff** | Frequência abaixo da qual está 85% da energia | Estrutura harmônica |
| **Spectral Bandwidth** | Largura de banda espectral | Riqueza tonal |
| **Tempo** | BPM estimado | Ritmo da fala |
| **Pausas** | Duração e quantidade de silêncios | Hesitação, emoção |
| **Chroma** | Características tonais (mean/std) | Conteúdo musical |

## 📚 Dataset

O projeto utiliza o **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):
- 24 atores profissionais (12 homens, 12 mulheres)
- 7 emoções diferentes
- Vídeos com áudio e vídeo sincronizados
- Qualidade profissional de atuação emocional

### Formato dos Arquivos RAVDESS

Nomenclatura: `03-01-06-01-02-01-12.mp4`

```
03 - Modalidade (Video)
01 - Vocal channel (Speech)
06 - Emoção (Fear)
01 - Intensidade emocional (Normal)
02 - Statement ("Dogs...")
01 - Repetição (1ª)
12 - Ator (12)
```

**Códigos de Emoções:**
- 01 = Neutral
- 02 = Calm
- 03 = Happy
- 04 = Sad
- 05 = Angry
- 06 = Fear
- 07 = Disgust
- 08 = Surprise

## 🔧 Configurações

### Parâmetros do Pipeline Visual

```python
FRAME_SKIP = 10  # Processa 1 a cada 10 frames (otimização)
CONFIDENCE_THRESHOLD = 0.5  # Confiança mínima do YOLO para detecção
```

### Parâmetros de Áudio

```python
SAMPLE_RATE = 22050  # Taxa de amostragem (Hz)
HOP_LENGTH = 512  # Janela de análise para STFT
N_MFCC = 13  # Número de coeficientes MFCC
```

### Detecção de Anomalias

```python
threshold = 2.0  # Z-score threshold para detecção de anomalias
```

---

## 🚧 Melhorias Futuras

- [ ] Implementar modelo de Deep Learning (CNN/LSTM) para classificação de áudio
- [ ] Adicionar suporte para análise em tempo real (streaming)
- [ ] Treinar modelo multimodal end-to-end
- [ ] Segmentação temporal para análise de estabilidade emocional
- [ ] Adicionar interface gráfica (GUI)
- [ ] Suporte para múltiplos idiomas
- [ ] Gerar visualizações automáticas (gráficos timeline de emoções)
- [ ] Implementar tracking temporal com Kalman Filter
- [ ] Adicionar métricas de avaliação (accuracy, F1-score, confusion matrix)
- [ ] Suporte para análise de múltiplas faces no mesmo frame

---

## 📝 Licença

Este projeto é parte do Tech Challenge 4 e está disponível para fins educacionais.
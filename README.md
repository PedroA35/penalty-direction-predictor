# ⚽ Penalty Kick Direction Predictor (Pose-Based)

Previsão da direção de remate em penáltis de futebol (esquerda / centro / direita, do ponto de vista do guarda-redes) a partir da biomecânica e pose corporal do marcador **antes do contacto com a bola**.

---

## 🎯 Objetivo & Abordagem

Ao contrário de modelos convencionais baseados em redes convolucionais 3D pesadas aplicadas a pixels brutos, este projeto utiliza uma abordagem focada em **biomecânica e interpretabilidade**:
1. **Pose Estimation (YOLO-Pose):** Extrai as coordenadas temporais das principais articulações (ancas, ombros, joelhos, tornozelos e pés).
2. **Engenharia de Atributos Biomecânicos:** Ângulos de rotação do tronco, bacia, inclinação da perna de apoio e orientação relativa antes do remate.
3. **Machine Learning Tabular:** Random Forest / XGBoost com cortes temporais retrospetivos (ex: 25 frames antes, 15 frames antes, etc.).
4. **Validação Rigorosa:** Avaliação por `GroupKFold` (agrupada por jogador) para evitar sobreajuste à forma de correr de um atleta específico.

---

## 📁 Estrutura do Repositório

```text
PENALTY-ANALYSIS/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── labels.csv           # Anotações de clips, links e timestamps
│   ├── raw/                 # Ficheiros brutos descarregados (ignorado no git)
│   ├── clips/               # Clips curtos segmentados (ignorado no git)
│   └── processed/           # Coordenadas articulares extraídas (ignorado no git)
├── src/
│   ├── __init__.py
│   ├── download.py          # Download cirúrgico com yt-dlp + ffmpeg
│   ├── pose.py              # Deteção do marcador e extração de landmarks
│   ├── features.py          # Engenharia de atributos e normalização corporal
│   └── train.py             # Validação por jogador e treino dos modelos
└── app/
    └── app.py               # Interface gráfica interativa (Streamlit)
```

---

## 🚀 Como Começar

### 1. Criar e Ativar o Ambiente Virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Anotar Penáltis em `data/labels.csv`

Adiciona os teus penáltis no ficheiro `data/labels.csv`:
```csv
clip_id,jogador,pe_remate,direcao,notas
clip_0001,Kylian Mbappe,D,esquerda,final_2022
clip_0002,Lionel Messi,E,direita,final_2022
```
* `clip_id`: Corresponde ao ficheiro de vídeo em `data/clips/<clip_id>.mp4`.
* `pe_remate`: `D` (direito) ou `E` (esquerdo).
* `direcao`: `esquerda`, `centro` ou `direita` (sempre do ponto de vista do guarda-redes).

### 3. Extrair e Validar Visualmente a Pose

Coloca os vídeos cortados na pasta `data/clips/` (ex: `clip_0001.mp4`, `clip_0002.mp4`) e corre:

```bash
python src/pose.py
```
Gera ficheiros com o esqueleto sobreposto em `data/clips/<clip_id>_annotated.mp4` para confirmação visual da deteção do marcador e guarda as coordenadas articulares em `data/processed/`.


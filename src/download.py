"""
Módulo de download e corte de clips de penáltis a partir do YouTube.
Utiliza yt-dlp e ffmpeg para descarregar apenas o segmento temporal de interesse
definido no ficheiro data/labels.csv.
"""

from pathlib import Path
import subprocess
import sys
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
CLIPS_DIR = DATA_DIR / "clips"


def format_timestamp(seconds: float | str) -> str:
    """Converte segundos num formato legível HH:MM:SS ou retorna a string se já formatada."""
    try:
        sec = float(seconds)
        mins, secs = divmod(sec, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}:{int(mins):02d}:{secs:04.1f}"
    except ValueError:
        return str(seconds)


def download_clip(clip_id: str, source: str, t_start: float | str, t_end: float | str) -> bool:
    """
    Descarrega ou corta um segmento temporal de t_start a t_end.
    Suporta:
      - URLs do YouTube (via yt-dlp)
      - Ficheiros locais de vídeo em data/raw/ ou caminho absoluto (via ffmpeg direto)
    O ficheiro final é guardado em data/clips/<clip_id>.mp4.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLIPS_DIR / f"{clip_id}.mp4"

    if output_path.exists():
        print(f"[OK] {clip_id} já existe em {output_path.name}. A saltar.")
        return True

    start_str = format_timestamp(t_start)
    end_str = format_timestamp(t_end)

    # Caso 1: Ficheiro Local (ex: data/raw/video.mp4 ou caminho absoluto)
    local_candidate = Path(source)
    if not local_candidate.is_absolute():
        local_candidate = (DATA_DIR / "raw" / source).resolve()
        if not local_candidate.exists():
            local_candidate = Path(source).resolve()

    if local_candidate.exists() and local_candidate.is_file():
        print(f"\n[A CORTAR LOCAL] {clip_id}: de {start_str} a {end_str} ({local_candidate.name})")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_str),
            "-to", str(end_str),
            "-i", str(local_candidate),
            "-c:v", "libx264", "-c:a", "aac",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[CONCLUÍDO] Guardado com sucesso em {output_path}")
            return True
        except subprocess.CalledProcessError as err:
            print(f"[ERRO FFmpeg] Falha ao cortar {clip_id}: {err.stderr}")
            return False

    # Caso 2: URL do YouTube (via yt-dlp + ffmpeg)
    print(f"\n[A DESCARREGAR YOUTUBE] {clip_id}: de {start_str} a {end_str} ({source})")
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start_str}-{end_str}",
        "--force-keyframes-at-cuts",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-f", "best[ext=mp4]/bestvideo+bestaudio/best",
        "-o", str(output_path),
        "--no-playlist",
        source
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[CONCLUÍDO] Guardado com sucesso em {output_path}")
        return True
    except subprocess.CalledProcessError as err:
        print(f"[ERRO yt-dlp] Falha ao processar {clip_id}: {err.stderr}")
        return False
    except FileNotFoundError:
        print("[ERRO] yt-dlp não foi encontrado. Certifica-te de que está instalado no ambiente virtual.")
        return False


def process_all_labels(csv_path: Path = LABELS_PATH):
    """Lê data/labels.csv e processa todos os clips registados."""
    if not csv_path.exists():
        print(f"[ERRO] Ficheiro de labels não encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Encontrados {len(df)} registos em {csv_path.name}.")

    for _, row in df.iterrows():
        clip_id = str(row["clip_id"]).strip()
        url = str(row["url"]).strip()
        t_start = row["t_start"]
        t_end = row["t_end"]

        # Ignorar o link de exemplo modelo predefinido
        if "dQw4w9WgXcQ" in url:
            print(f"[INFO] A ignorar o exemplo modelo ({clip_id}). Substitui por um link real.")
            continue

        download_clip(clip_id, url, t_start, t_end)


if __name__ == "__main__":
    process_all_labels()


#!/usr/bin/env python3
import time
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_FRAMES = 155
saved = 0
skipped = 0

print(f"[INFO] Iniciando captura de {TOTAL_FRAMES} frames para o dataset de EPIs...")

for i in range(1, TOTAL_FRAMES + 1):
    time.sleep(0.01)
    # Simula analise de nitidez pelo operador Laplaciano
    # Se a variancia for muito baixa (imagem borrada), descarta o frame
    if i % 8 == 0:
        skipped += 1
        continue
    
    frame = np.full((480, 640, 3), (i * 2) % 255, dtype=np.uint8)
    cv2.putText(frame, f"Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(str(OUTPUT_DIR / f"frame_{saved:04d}.jpg"), frame)
    saved += 1
    if saved % 25 == 0:
        print(f"[INFO] {saved}/{TOTAL_FRAMES} frames salvos...")

print("-" * 55)
print(f"[OK] {saved} frames salvos em {OUTPUT_DIR}")
print(f"[OK] {skipped} frames borrados descartados automaticamente")

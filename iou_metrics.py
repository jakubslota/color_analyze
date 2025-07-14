import os
import cv2
import pandas as pd
import numpy as np


def compute_iou_masks(pred_mask, gt_mask):
    """
    Oblicza IOU pomiędzy maską predykcyjną a referencyjną.
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


# Ścieżki
mask_auto_dir = 'maski2/potwierdzone'  # Folder z wygenerowanymi maskami lub wynikami segmentacji
mask_reference_dir = 'wybrane zdj/potwierdzone_maski'  # Folder z maskami referencyjnymi
output_excel = 'iou_wyniki_potwierdzone.xlsx'  # Wynikowy plik Excel

# Lista wyników
wyniki = []

# Pobierz listę plików
mask_files = [f for f in os.listdir(mask_auto_dir) if f.lower().endswith('.bmp')]

for mask_file in sorted(mask_files):
    # Dopasuj nazwę maski referencyjnej
    nazwa_bez = os.path.splitext(mask_file)[0].split('_')[0]  # np. IMD001
    mask_ref_name = nazwa_bez + '_lesion.bmp'

    mask_auto_path = os.path.join(mask_auto_dir, mask_file)
    mask_ref_path = os.path.join(mask_reference_dir, mask_ref_name)

    # Wczytaj maski
    mask_auto = cv2.imread(mask_auto_path, cv2.IMREAD_GRAYSCALE)
    mask_ref = cv2.imread(mask_ref_path, cv2.IMREAD_GRAYSCALE)

    if mask_auto is None or mask_ref is None:
        print(f"❌ Błąd w pliku: {mask_file}")
        continue

    # Oblicz IOU
    iou = compute_iou_masks(mask_auto, mask_ref)

    # Zapisz wynik
    wyniki.append([mask_file, mask_ref_name, iou])
    print(f"{mask_file} - IOU: {iou:.4f}")

# Zapis do Excela
df = pd.DataFrame(wyniki, columns=["Maska_wygenerowana", "Maska_referencyjna", "IOU"])
df.to_excel(output_excel, index=False)

print(f"\n✅ Wyniki zapisane do: {output_excel}")

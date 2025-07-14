import os
import cv2
import matplotlib.pyplot as plt
import preprocessing as ps

# # Ścieżki do folderów
# image_dir = 'wybrane zdj/zdrowe'            # Należy podać poprawną ścieżkę do folderu ze zdjęciami
# mask_dir = 'wybrane zdj/zdrowe_maski'       # Należy podać poprawną ścieżkę do folderu z maskami
# output_dir = 'maski/zdrowe'                 # Należy podać poprawną ścieżkę do folderu wyjściowego
#
# # Utwórz folder wyjściowy, jeśli nie istnieje
# os.makedirs(output_dir, exist_ok=True)
#
# # Pobierz listę wszystkich plików BMP
# image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]
#
# # Przejdź po każdym pliku
# for image_file in sorted(image_files):
#     print(f"\n Przetwarzanie pliku: {image_file}")
#
#     image_path = os.path.join(image_dir, image_file)
#     mask_file = os.path.splitext(image_file)[0] + '_lesion.bmp'
#     mask_path = os.path.join(mask_dir, mask_file)
#
#     # Wczytanie obrazu
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Nie udało się wczytać obrazu.")
#         continue
#
#     # Wczytanie maski referencyjnej
#     mask_reference = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask_reference is None:
#         print("Nie znaleziono maski referencyjnej:", mask_path)
#         continue
#
#     # Ekstrakcja zmiany i maski automatycznej
#     try:
#         lesion, mask_auto, largest_contour = ps.extract_lesion(image)
#     except Exception as e:
#         print("Błąd w extract_lesion:", e)
#         continue
#
#     # Wizualizacja i zapis
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(image_rgb)
#     plt.title("Oryginalny obraz")
#     plt.axis('off')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(mask_auto, cmap='gray')
#     plt.title("Maska wygenerowana")
#     plt.axis('off')
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(mask_reference, cmap='gray')
#     plt.title("Maska referencyjna")
#     plt.axis('off')
#
#     plt.tight_layout()
#
#     # Zapis do pliku
#     output_file = os.path.splitext(image_file)[0] + '_porownanie.png'
#     output_path = os.path.join(output_dir, output_file)
#     plt.savefig(output_path)
#     plt.close()
#
#     print(f"Zapisano wizualizację do: {output_path}")

# Ścieżki do folderów
image_dir = 'wybrane zdj/potwierdzone'            # Folder ze zdjęciami
output_dir = 'maski2/potwierdzone'                 # Folder do zapisu masek wygenerowanych

# Utwórz folder wyjściowy jeśli nie istnieje
os.makedirs(output_dir, exist_ok=True)

# Pobierz listę wszystkich plików BMP
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]

# Przejdź po każdym pliku
for image_file in sorted(image_files):
    print(f"\nPrzetwarzanie pliku: {image_file}")

    image_path = os.path.join(image_dir, image_file)

    # Wczytanie obrazu
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Nie udało się wczytać obrazu.")
        continue

    # Ekstrakcja zmiany i maski automatycznej
    try:
        lesion, mask_auto, largest_contour = ps.extract_lesion(image)
    except Exception as e:
        print(f"❌ Błąd w extract_lesion: {e}")
        continue

    # Przygotowanie nazwy pliku wynikowego
    output_file = os.path.splitext(image_file)[0] + '_mask_auto.bmp'
    output_path = os.path.join(output_dir, output_file)

    # Zapis maski jako plik BMP
    cv2.imwrite(output_path, mask_auto)

    print(f"✅ Zapisano maskę do: {output_path}")

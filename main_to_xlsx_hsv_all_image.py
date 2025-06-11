import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import color_hsv as color
import preprocessing as ps

# Ścieżki
folder_wejsciowy = r"D:\STUDIA MAGISTERSKIE\python\color_analyze\wybrane zdj\zdrowe"
folder_wyjsciowy = r"D:\STUDIA MAGISTERSKIE\python\color_analyze\wyniki_hsv\zdrowe"
plik_wyjsciowy = "analiza_koloru_hsv_zdrowe.xlsx"

# Tworzenie folderu wyjściowego
os.makedirs(folder_wyjsciowy, exist_ok=True)

# Kolory
kolory = ["white", "black", "red", "blue_gray", "light_brown", "dark_brown"]
naglowki = ["nazwa_pliku", "wersja", "piksele_znamienia", "c_value", "kolory_C"] + kolory
wiersze = []

# Warianty analiz
analizy = {
    "Wariant_1": color.analyze_by_reference_hsv,
    "Wariant_2": color.analyze_by_reference_hsv_pixel_to_pixel,
    "Wariant_3": color.analyze_by_cluster_centers
}

# Przetwarzanie zdjęć
for nazwa_pliku in os.listdir(folder_wejsciowy):
    if nazwa_pliku.lower().endswith(".bmp") and '_' not in nazwa_pliku:
        sciezka = os.path.join(folder_wejsciowy, nazwa_pliku)
        print(f"\n Przetwarzam: {nazwa_pliku}")
        obraz = cv2.imread(sciezka)
        if obraz is None:
            print(f"Nie można otworzyć: {nazwa_pliku}")
            continue

        try:
            lesion, mask, _ = ps.extract_lesion(obraz)
            piksele_znamienia = int(cv2.countNonZero(mask))

            for wersja_nazwa, funkcja in analizy.items():
                print(f"Analiza: {wersja_nazwa}")
                nazwa_bez_rozszerzenia = os.path.splitext(nazwa_pliku)[0]
                wynikowa_nazwa = f"{nazwa_bez_rozszerzenia}_{wersja_nazwa}.png"
                save_path = os.path.join(folder_wyjsciowy, wynikowa_nazwa)

                # Wywołanie analizy z obrazem wynikowym
                if "n_clusters" in funkcja.__code__.co_varnames:
                    c_value, included_colors, color_counter, label_image = funkcja(
                        obraz, mask, n_clusters=11, save_path=None, return_image=True)
                else:
                    c_value, included_colors, color_counter, label_image = funkcja(
                        obraz, mask, save_path=None, return_image=True)

                # Wizualizacja: oryginalny + etykietowany
                obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(obraz_rgb)
                plt.title("Oryginalny obraz")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(label_image)
                plt.title("Obraz kolorystyczny")
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                print(f"Zapisano obraz do: {save_path}")

                # Dane do Excela
                kolory_C = ";".join(included_colors)
                wartosci_kolorow = [int(color_counter.get(kolor, 0)) for kolor in kolory]
                wiersz = [nazwa_pliku, wersja_nazwa, piksele_znamienia, c_value, kolory_C] + wartosci_kolorow
                wiersze.append(wiersz)

        except Exception as e:
            print(f"Błąd w pliku {nazwa_pliku}: {e}")

# Zapis danych do Excela
df = pd.DataFrame(wiersze, columns=naglowki)
df.to_excel(plik_wyjsciowy, index=False)
print(f"\n Wyniki zapisane do pliku: {plik_wyjsciowy}")

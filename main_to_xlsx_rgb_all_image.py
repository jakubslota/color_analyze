import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import color_rgb as color
import preprocessing as ps

# Ścieżki
folder_wejsciowy = r"D:\STUDIA MAGISTERSKIE\python\color_analyze\wybrane zdj\zdrowe"
folder_wyjsciowy = r"D:\STUDIA MAGISTERSKIE\python\color_analyze\wyniki_rgb\zdrowe"
plik_wyjsciowy = "analiza_koloru_rgb_zdrowe.xlsx"

# Tworzenie folderu na wyniki
os.makedirs(folder_wyjsciowy, exist_ok=True)

# Kolory do raportowania
kolory = ["white", "black", "red", "blue_gray", "light_brown", "dark_brown"]
naglowki = ["nazwa_pliku", "wersja", "piksele_znamienia", "c_value", "kolory_C"] + kolory
wiersze = []

# Wersje analiz
analizy = {
    "Wariant_1": color.analyze_by_reference_rgb,
    "Wariant_2": color.analyze_by_reference_rgb_pixel_to_pixel,
    "Wariant_3": color.analyze_by_reference_rgb_clustered
}

# Przetwarzanie zdjęć
for nazwa_pliku in os.listdir(folder_wejsciowy):
    if nazwa_pliku.lower().endswith(".bmp") and "_" not in nazwa_pliku:
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
                print(f" Analiza: {wersja_nazwa}")

                # Ścieżka do zapisu obrazu
                nazwa_bez_rozszerzenia = os.path.splitext(nazwa_pliku)[0]
                nazwa_zdjecia = f"{nazwa_bez_rozszerzenia}_{wersja_nazwa}.png"
                sciezka_zapisu = os.path.join(folder_wyjsciowy, nazwa_zdjecia)

                # Wykonaj analizę i zbierz wynik oraz obraz
                if "n_clusters" in funkcja.__code__.co_varnames:
                    c_value, included_colors, color_counter, wynikowy_obraz = funkcja(obraz, mask, n_clusters=11, save_path=None, return_image=True)
                else:
                    c_value, included_colors, color_counter, wynikowy_obraz = funkcja(obraz, mask, save_path=None, return_image=True)

                obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(obraz_rgb)
                plt.title("Oryginalny obraz")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(wynikowy_obraz)
                plt.title("Obraz kolorystyczny")
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(sciezka_zapisu)
                plt.close()

                # Dane do Excela
                kolory_C = ";".join(included_colors)
                wartosci_kolorow = [int(color_counter.get(kolor, 0)) for kolor in kolory]
                wiersz = [nazwa_pliku, wersja_nazwa, piksele_znamienia, c_value, kolory_C] + wartosci_kolorow
                wiersze.append(wiersz)

        except Exception as e:
            print(f"Błąd w pliku {nazwa_pliku}: {e}")

# Zapis wyników do Excela
df = pd.DataFrame(wiersze, columns=naglowki)
df.to_excel(plik_wyjsciowy, index=False)
print(f"\n Wyniki zapisane do pliku: {plik_wyjsciowy}")

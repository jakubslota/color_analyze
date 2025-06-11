
import cv2
import preprocessing as ps
import color_rgb as color

# Wczytanie obrazu
image_path = 'wybrane zdj/potwierdzone/IMD211.bmp'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Nie znaleziono obrazu. Sprawdź ścieżkę do pliku.")

lesion, mask, largest_contour = ps.extract_lesion(image)

print()
print("1 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_reference_rgb(image, mask, n_clusters=11)

print()
print("2 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_reference_rgb_pixel_to_pixel(image, mask, debug_samples=10)

print()
print("3 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_reference_rgb_clustered(image, mask, n_clusters=11)




import cv2
import preprocessing as ps
import color_hsv as color

# Wczytanie obrazu
image_path = 'wybrane zdj/potwierdzone/IMD437.bmp'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Nie znaleziono obrazu. Sprawdź ścieżkę do pliku.")

lesion, mask, largest_contour = ps.extract_lesion(image)

print()
print("1 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_reference_hsv(image, mask, n_clusters=11)

print()
print("2 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_reference_hsv_pixel_to_pixel(image, mask, n_clusters=11)

print()
print("3 WERSJA")
print()

c_value, color_counts, counter = color.analyze_by_cluster_centers(image, mask, n_clusters=11)



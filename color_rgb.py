import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import random

# 1 WERSJA KMeans bez zakresów kolorów tylko pojedyncze kolory RGB

def analyze_by_reference_rgb(image, mask, n_clusters=6, save_path=None, return_image=False):
    # Dopasowanie maski do obrazu
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ekstrakcja pikseli ze znamienia (RGB)
    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                b, g, r = image[y, x]
                pixels.append([r, g, b])
                coords.append((y, x))
    pixels = np.array(pixels)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Referencyjne kolory z tabeli (RGB)
    reference_colors = {
        "light_brown": np.array([200, 155, 130]),
        "red": np.array([255, 0, 0]),
        "dark_brown": np.array([126, 67, 48]),
        "white": np.array([230, 230, 230]),
        "black": np.array([31, 26, 26]),
        "blue_gray": np.array([75, 112, 137])
    }

    color_map = {
        "light_brown": (200, 155, 130),
        "red": (200, 50, 50),
        "dark_brown": (126, 67, 48),
        "white": (230, 230, 230),
        "black": (31, 26, 26),
        "blue_gray": (75, 112, 137)
    }

    # odległość euklidesowa do najbliższego koloru referencyjnego
    def match_reference_color1(rgb):
        distances = {name: np.linalg.norm(rgb - ref) for name, ref in reference_colors.items()}
        return min(distances, key=distances.get)

    # Obraz etykiet
    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []
    for idx, (y, x) in enumerate(coords):
        rgb_center = centers[labels[idx]]
        category = match_reference_color1(rgb_center)
        category_labels.append(category)
        label_image[y, x] = color_map[category]

    # Liczenie udziałów kolorów
    counter = Counter(category_labels)
    total_pixels = len(category_labels)
    included_colors = [cat for cat, count in counter.items() if (count / total_pixels) >= 0.15]
    C = len(included_colors) * 0.5

    # Wyniki
    print("Udział kolorów:")
    for cat, count in counter.items():
        print(f"  {cat}: {count} pikseli ({(count / total_pixels) * 100:.2f}%)")
    print(f"\n Uwzględnione kolory (≥15%): {included_colors}")
    print(f"Wartość C: {C:.2f}")

    # Wizualizacja
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image)
    plt.title("Obszary wg kolorów referencyjnych")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        save_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_image)
        print(f"Zapisano wynikowy obraz do: {save_path}")

    return C, included_colors, counter, label_image

# 2 WERSJA piksel po pikselu zakres kolorów RGB

kolory_rgb = {
    "light_brown": ((179, 149, 134), (255, 181, 77)),
    "dark_brown": ((102, 77, 77), (166, 69, 0)),
    "red": ((230, 23, 23), (255, 0, 0)),
    "black": ((0, 0, 0), (100, 100, 100)),
    "white": ((242, 242, 242), (255, 255, 255)),
    "blue_gray": ((57, 63, 64), (82, 102, 204))
}

color_map = {
    "light_brown": (200, 155, 130),
    "dark_brown": (126, 67, 48),
    "red": (200, 50, 50),
    "white": (230, 230, 230),
    "black": (31, 26, 26),
    "blue_gray": (75, 112, 137)
}

def in_rgb_range2(rgb, lower, upper):
    return all(lower[i] <= rgb[i] <= upper[i] for i in range(3))

def match_rgb_to_range_or_nearest2(rgb):
    for name, (lower, upper) in kolory_rgb.items():
        if in_rgb_range2(rgb, lower, upper):
            return name

    # Jeśli nie pasuje, przypisanie jest na podstawie najmniejszej odległości euklidesowej do środka zakresu
    min_dist = float('inf')
    closest = None
    for name, (lower, upper) in kolory_rgb.items():
        ref = [(l + u) / 2 for l, u in zip(lower, upper)]
        dist = np.linalg.norm(np.array(rgb) - np.array(ref))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

def analyze_by_reference_rgb_pixel_to_pixel(image, mask, n_clusters=11, debug_samples=10, save_path=None, return_image=False):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                b, g, r = image[y, x]
                pixels.append([r, g, b])
                coords.append((y, x))
    pixels = np.array(pixels)

    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []

    sample_indices = random.sample(range(len(coords)), min(debug_samples, len(coords)))
    print("\n DEBUG wybranych pikseli:")

    for idx, (y, x) in enumerate(coords):
        r, g, b = pixels[idx]
        category = match_rgb_to_range_or_nearest2((r, g, b))
        if idx in sample_indices:
            print(f"Piksel ({y}, {x}) | RGB = ({r}, {g}, {b}) → {category}")
        category_labels.append(category)
        label_image[y, x] = color_map[category]

    counter = Counter(category_labels)
    total_pixels = len(category_labels)
    included_colors = [cat for cat, count in counter.items() if (count / total_pixels) >= 0.15]
    C = len(included_colors) * 0.5

    print("\n Udział kolorów:")
    for cat, count in counter.items():
        print(f"  {cat}: {count} pikseli ({(count / total_pixels) * 100:.2f}%)")
    print(f"\n Uwzględnione kolory (≥15%): {included_colors}")
    print(f"Wartość C: {C:.2f}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image)
    plt.title("Klasyfikacja piksel po pikselu")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        save_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_image)
        print(f"Zapisano wynikowy obraz do: {save_path}")

    return C, included_colors, counter, label_image

# 3 WERSJA KMeans z zakresami kolorów RGB

kolory_rgb = {
    "light_brown": ((179, 149, 134), (255, 181, 77)),
    "dark_brown": ((102, 77, 77), (166, 69, 0)),
    "red": ((230, 23, 23), (255, 0, 0)),
    "black": ((0, 0, 0), (100, 100, 100)),
    "white": ((242, 242, 242), (255, 255, 255)),
    "blue_gray": ((57, 63, 64), (82, 102, 204))
}

color_map = {
    "light_brown": (200, 155, 130),
    "dark_brown": (126, 67, 48),
    "red": (200, 50, 50),
    "white": (230, 230, 230),
    "black": (31, 26, 26),
    "blue_gray": (75, 112, 137)
}

def in_rgb_range3(rgb, lower, upper):
    return all(lower[i] <= rgb[i] <= upper[i] for i in range(3))

def match_rgb_to_range_or_nearest3(rgb):
    for name, (lower, upper) in kolory_rgb.items():
        if in_rgb_range3(rgb, lower, upper):
            return name

    # Odległość euklidesowa do środka zakresu
    min_dist = float('inf')
    closest = None
    for name, (lower, upper) in kolory_rgb.items():
        ref = [(l + u) / 2 for l, u in zip(lower, upper)]
        dist = np.linalg.norm(np.array(rgb) - np.array(ref))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

def analyze_by_reference_rgb_clustered(image, mask, n_clusters=11, debug_samples=10, save_path=None, return_image=False):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                b, g, r = image[y, x]
                pixels.append([r, g, b])
                coords.append((y, x))
    pixels = np.array(pixels)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []

    print("\n Środki klastrów (RGB):")
    cluster_categories = {}
    for i, center in enumerate(centers):
        category = match_rgb_to_range_or_nearest3(center)
        cluster_categories[i] = category
        print(f"  Klaster {i}: RGB = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) → {category}")

    sample_indices = random.sample(range(len(coords)), min(debug_samples, len(coords)))
    print("\n DEBUG wybranych pikseli:")

    for idx, (y, x) in enumerate(coords):
        cluster_idx = labels[idx]
        category = cluster_categories[cluster_idx]
        category_labels.append(category)
        label_image[y, x] = color_map[category]

        if idx in sample_indices:
            r, g, b = pixels[idx]
            center = centers[cluster_idx]
            print(f"Piksel ({y}, {x}) | RGB = ({r}, {g}, {b})")
            print(f"Należy do klastra {cluster_idx} → środek: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
            print(f"Przypisany kolor: {category}\n")

    counter = Counter(category_labels)
    total_pixels = len(category_labels)
    included_colors = [cat for cat, count in counter.items() if (count / total_pixels) >= 0.15]
    C = len(included_colors) * 0.5

    print("\n Udział kolorów:")
    for cat, count in counter.items():
        print(f"  {cat}: {count} pikseli ({(count / total_pixels) * 100:.2f}%)")
    print(f"\n Uwzględnione kolory (≥15%): {included_colors}")
    print(f"Wartość C: {C:.2f}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image)
    plt.title("klastrowanie Kmeans")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        save_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_image)
        print(f"Zapisano wynikowy obraz do: {save_path}")

    return C, included_colors, counter, label_image



import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from skimage.color import rgb2hsv
import random

# 1 WERSJA kolory rgb zamienia na hsv, pojedyncze wartosci kolorów z wykorzystaniem KMeans

def analyze_by_reference_hsv(image, mask, n_clusters=11, save_path=None, return_image=False):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = rgb2hsv(image_rgb)

    ref_rgb = {
        "light_brown": (200, 155, 130),
        "red": (255, 0, 0),
        "dark_brown": (126, 67, 48),
        "white": (230, 230, 230),
        "black": (31, 26, 26),
        "blue_gray": (75, 112, 137)
    }

    ref_hsv = {name: rgb2hsv(np.array([[np.array(rgb) / 255.0]])).squeeze() for name, rgb in ref_rgb.items()}

    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                h, s, v = image_hsv[y, x]
                pixels.append([h * 360, s * 100, v * 100])
                coords.append((y, x))
    pixels = np.array(pixels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    def match_reference_hsv(hsv_val):
        h, s, v = hsv_val
        hsv_ref_scaled = {name: (ref[0]*360, ref[1]*100, ref[2]*100) for name, ref in ref_hsv.items()}
        distances = {name: np.linalg.norm(np.array([h, s, v]) - np.array(hsv_scaled))
                     for name, hsv_scaled in hsv_ref_scaled.items()}
        return min(distances, key=distances.get)

    color_map = {
        "light_brown": (200, 155, 130),
        "red": (255, 0, 0),
        "dark_brown": (126, 67, 48),
        "white": (230, 230, 230),
        "black": (31, 26, 26),
        "blue_gray": (75, 112, 137)
    }

    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []
    for idx, (y, x) in enumerate(coords):
        h, s, v = centers[labels[idx]]
        category = match_reference_hsv((h, s, v))
        category_labels.append(category)
        label_image[y, x] = color_map[category]

    counter = Counter(category_labels)
    total_pixels = len(category_labels)
    included_colors = [cat for cat, count in counter.items() if (count / total_pixels) >= 0.15]
    C = len(included_colors) * 0.5

    print("Udział kolorów:")
    for cat, count in counter.items():
        print(f"  {cat}: {count} pikseli ({(count / total_pixels) * 100:.2f}%)")
    print(f"\n Uwzględnione kolory (≥15%): {included_colors}")
    print(f"Wartość C: {C:.2f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image)
    plt.title("Wariant 1: Klasyfikacja kolorów HSV")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        save_img = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    if return_image:
        return C, included_colors, counter, label_image
    else:
        return C, included_colors, counter



# 2 WERSJA piksel po pikselu przypisanie, zakres kolorów

# Zakresy HSV (H: 0–360, S: 0–100, V: 0–100)
color_ranges_hsv = {
    "light_brown": [[20, 35], [25, 70], [70, 100]],
    "dark_brown": [[0, 25], [25, 100], [40, 65]],
    "red": [[[0, 1], [359, 360]], [99, 100], [99, 100]],
    "white": [[0, 360], [0, 5], [95, 100]],
    "black": [[0, 360], [0, 100], [0, 40]],
    "blue_gray": [[190, 230], [10, 60], [25, 80]]
}

# Kolory RGB do wizualizacji
color_map = {
    "light_brown": (200, 155, 130),
    "dark_brown": (126, 67, 48),
    "red": (255, 0, 0),
    "white": (230, 230, 230),
    "black": (31, 26, 26),
    "blue_gray": (75, 112, 137)
}

def in_range(val, min_val, max_val):
    return min_val <= val <= max_val

# Funkcja do dopasowania koloru HSV do zakresów
def match_reference_hsv_force1(hsv_val, debug=False):
    h, s, v = hsv_val
    min_distance = float("inf")
    best_match = None
    debug_candidates = []

    for name, (h_range_all, s_range, v_range) in color_ranges_hsv.items():
        # Obsługa wielozakresowego H dla koloru czerwonego
        if isinstance(h_range_all[0], list):
            h_match = any(in_range(h, *hr) for hr in h_range_all)
            h_refs = [(hr[0] + hr[1]) / 2 for hr in h_range_all]
            h_ref = min(h_refs, key=lambda ref: abs(h - ref))
        else:
            h_match = in_range(h, *h_range_all)
            h_ref = (h_range_all[0] + h_range_all[1]) / 2

        s_ref = (s_range[0] + s_range[1]) / 2
        v_ref = (v_range[0] + v_range[1]) / 2

        dist = np.linalg.norm(np.array([h, s, v]) - np.array([h_ref, s_ref, v_ref]))
        debug_candidates.append((name, dist))

        if h_match and in_range(s, *s_range) and in_range(v, *v_range):
            return name

        if dist < min_distance:
            min_distance = dist
            best_match = name

    if debug:
        print(f"\n HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")
        for name, dist in sorted(debug_candidates, key=lambda x: x[1]):
            print(f"{name}: dystans = {dist:.2f}")
        print(f"Przypisano do: {best_match} (najbliższy kolor)\n")

    return best_match

def analyze_by_reference_hsv_pixel_to_pixel(image, mask, n_clusters=11, save_path=None, return_image=False):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = rgb2hsv(image_rgb)

    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                h, s, v = image_hsv[y, x]
                pixels.append([h * 360, s * 100, v * 100])
                coords.append((y, x))
    pixels = np.array(pixels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []

    print("\n Środki klastrów (HSV):")
    for i, (h, s, v) in enumerate(centers):
        print(f"Klaster {i}: H={h:.1f}, S={s:.1f}, V={v:.1f}")

    for idx, (y, x) in enumerate(coords):
        h, s, v = pixels[idx]
        category = match_reference_hsv_force1((h, s, v), debug=False)
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

    if return_image:
        return C, included_colors, counter, label_image
    else:
        return C, included_colors, counter


# 3 WERSJA za pomoca klastrowania KMeans z priorytetem na kolor czarny, zakresy kolorów

# Zakresy HSV (H: 0–360, S: 0–100, V: 0–100)
color_ranges_hsv = {
    "light_brown": [[20, 35], [25, 70], [70, 100]],
    "dark_brown": [[0, 25], [25, 100], [40, 65]],
    "red": [[[0, 1], [359, 360]], [99, 100], [99, 100]],
    "white": [[0, 360], [0, 5], [95, 100]],
    "black": [[0, 360], [0, 100], [0, 40]],
    "blue_gray": [[190, 230], [10, 60], [25, 80]]
}

color_map = {
    "light_brown": (200, 155, 130),
    "dark_brown": (126, 67, 48),
    "red": (255, 0, 0),
    "white": (230, 230, 230),
    "black": (31, 26, 26),
    "blue_gray": (75, 112, 137)
}

def in_range(val, min_val, max_val):
    return min_val <= val <= max_val

def match_reference_hsv_force2(hsv_val):
    h, s, v = hsv_val

    # Najpierw sprawdzamy czarny kolor
    h_range_all, s_range, v_range = color_ranges_hsv["black"]
    if in_range(h, *h_range_all) and in_range(s, *s_range) and in_range(v, *v_range):
        return "black"

    min_distance = float("inf")
    best_match = None

    for name, (h_range_all, s_range, v_range) in color_ranges_hsv.items():
        if name == "black":
            continue

        if isinstance(h_range_all[0], list):
            h_refs = [(hr[0] + hr[1]) / 2 for hr in h_range_all]
            h_ref = min(h_refs, key=lambda ref: abs(h - ref))
        else:
            h_ref = (h_range_all[0] + h_range_all[1]) / 2

        s_ref = (s_range[0] + s_range[1]) / 2
        v_ref = (v_range[0] + v_range[1]) / 2

        dist = np.linalg.norm(np.array([h, s, v]) - np.array([h_ref, s_ref, v_ref]))
        if dist < min_distance:
            min_distance = dist
            best_match = name

    return best_match

def analyze_by_cluster_centers(image, mask, n_clusters=11, debug_samples=1, save_path=None, return_image=False):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = rgb2hsv(image_rgb)

    pixels = []
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] > 0:
                h, s, v = image_hsv[y, x]
                pixels.append([h * 360, s * 100, v * 100])
                coords.append((y, x))
    pixels = np.array(pixels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    label_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    category_labels = []

    print("\n Środki klastrów (HSV):")
    cluster_categories = {}
    for i, center in enumerate(centers):
        h, s, v = center
        category = match_reference_hsv_force2((h, s, v))
        cluster_categories[i] = category
        print(f"Klaster {i}: H={h:.1f}, S={s:.1f}, V={v:.1f} → {category}")

    # Debugowanie losowych pikseli
    sample_indices = random.sample(range(len(coords)), min(debug_samples, len(coords)))
    print("\n DEBUG wybranych pikseli:")
    for idx in sample_indices:
        y, x = coords[idx]
        h, s, v = pixels[idx]
        cluster_idx = labels[idx]
        c_h, c_s, c_v = centers[cluster_idx]
        category = cluster_categories[cluster_idx]
        print(f"Piksel ({y}, {x}) | HSV = ({h:.1f}, {s:.1f}, {v:.1f})")
        print(f"Należy do klastra {cluster_idx} → środek: ({c_h:.1f}, {c_s:.1f}, {c_v:.1f})")
        print(f"Przypisany kolor: {category}\n")

    for idx, (y, x) in enumerate(coords):
        cluster_idx = labels[idx]
        category = cluster_categories[cluster_idx]
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

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Oryginalny obraz")

    plt.subplot(1, 2, 2)
    plt.imshow(label_image)
    plt.title("Klastrowanie KMeans z priorytetem na kolor czarny")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if save_path:
        save_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_image)

    if return_image:
        return C, included_colors, counter, label_image
    else:
        return C, included_colors, counter






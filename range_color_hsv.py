import matplotlib.pyplot as plt
import colorsys

# Zakresy kolorów w przestrzeni HSV
color_ranges_hsv = {
    "light_brown": [[20, 35], [25, 70], [70, 100]],
    "dark_brown": [[0, 25], [25, 100], [40, 65]],
    "red1": [[0, 1], [99, 100], [99, 100]],
    "red2": [[359, 360], [99, 100], [99, 100]],
    "white": [[0, 360], [0, 5], [95, 100]],
    "black": [[0, 360], [0, 100], [0, 40]],
    "blue_gray": [[190, 230], [10, 60], [25, 80]]
}


def hsv_to_rgb_patch(h, s, v):
    """HSV [0–360, 0–100, 0–100] → RGB [0–1]"""
    return colorsys.hsv_to_rgb(h/360, s/100, v/100)

# Tworzenie wizualizacji
n = len(color_ranges_hsv)
fig, axes = plt.subplots(2, n, figsize=(n * 2, 3))
fig.suptitle("Zakresy HSV – dolna (góra) i górna (dół) granica", fontsize=14)

for i, (name, (h_range, s_range, v_range)) in enumerate(color_ranges_hsv.items()):
    # Dolna granica
    h_low, s_low, v_low = h_range[0], s_range[0], v_range[0]
    rgb_low = hsv_to_rgb_patch(h_low, s_low, v_low)
    axes[0, i].imshow([[rgb_low]])
    axes[0, i].set_title(name, fontsize=10)
    axes[0, i].axis('off')

    # Górna granica
    h_up, s_up, v_up = h_range[1], s_range[1], v_range[1]
    rgb_up = hsv_to_rgb_patch(h_up, s_up, v_up)
    axes[1, i].imshow([[rgb_up]])
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

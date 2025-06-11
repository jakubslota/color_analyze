from PIL import Image, ImageDraw, ImageFont

# Zakresy RGB – (lower, upper)
kolory_rgb = {
    "light_brown": ((179, 149, 134), (255, 181, 77)),
    "dark_brown": ((102, 77, 77), (166, 69, 0)),
    "red": ((230, 23, 23), (255, 0, 0)),
    "black": ((0, 0, 0), (100, 100, 100)),
    "white": ((242, 242, 242), (255, 255, 255)),
    "blue_gray": ((57, 63, 64), (82, 102, 204))
}

# Parametry obrazu
square_size = 100
label_height = 30
cols = len(kolory_rgb)
rows = 2

# Tworzymy obraz z miejscem na etykiety
width = cols * square_size
height = rows * square_size + label_height
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for i, (nazwa, (lower, upper)) in enumerate(kolory_rgb.items()):
    x = i * square_size

    # Górny prostokąt – lower
    draw.rectangle([x, 0, x + square_size, square_size], fill=lower)

    # Dolny prostokąt – upper
    draw.rectangle([x, square_size, x + square_size, 2 * square_size], fill=upper)

    # Obliczenie szerokości i wysokości napisu przy użyciu textbbox
    bbox = draw.textbbox((0, 0), nazwa, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Pozycja etykiety
    text_x = x + (square_size - text_width) // 2
    text_y = 2 * square_size + 5
    draw.text((text_x, text_y), nazwa, fill="black", font=font)

# Wyświetlenie obrazu
img.show()

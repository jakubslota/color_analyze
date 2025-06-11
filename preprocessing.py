import cv2
import numpy as np

def extract_lesion(image):
    # Konwersja na skalę szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rozmycie Gaussa
    blurred = cv2.GaussianBlur(gray, (29, 29), 0)

    # Progowanie metodą Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Wykrywanie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Usunięcie mniejszych konturów (mogą to być włosy lub szumy)
    for contour in contours:
        if cv2.contourArea(contour) < 10000:  # Próg do dostosowania
            cv2.drawContours(thresh, [contour], -1, 0, -1)

    # Ponowne wykrycie konturów po usunięciu mniejszych
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        # Usunięcie konturów dotykających krawędzi obrazu
        h, w = thresh.shape
        valid_contours = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                # Jeśli kontur dotyka którejkolwiek krawędzi, odrzuć go
                if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
                    break
            else:
                # Jeśli kontur nie dotyka krawędzi, dodaj go do listy
                valid_contours.append(contour)
    else:
        valid_contours = contours
    # Znalezienie największego konturu spośród ważnych
    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
    else:
        raise ValueError("Nie znaleziono odpowiedniego konturu dla znamienia.")

    # Tworzenie maski dla największego konturu
    lesion_mask = np.zeros_like(gray)
    cv2.drawContours(lesion_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Wyodrębnienie znamienia za pomocą maski
    extracted_lesion = cv2.bitwise_and(image, image, mask=lesion_mask)

    return extracted_lesion, lesion_mask, largest_contour

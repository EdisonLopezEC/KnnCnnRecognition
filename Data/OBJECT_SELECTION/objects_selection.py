import cv2
import os

def detect_and_extract_letters(image_path, min_area_percentage):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización para resaltar las letras
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcular el área total de la imagen
    total_area = image.shape[0] * image.shape[1]

    # Crear una carpeta para almacenar las imágenes extraídas
    output_folder = "./extract"
    os.makedirs(output_folder, exist_ok=True)

    index = 1  # Variable para el nombre de las subcarpetas

    # Crear una subcarpeta para esta ejecución del script
    subfolder_path = os.path.join(output_folder, f"placa{index}")
    os.makedirs(subfolder_path, exist_ok=True)

    # Crear una copia de la imagen original para visualización
    image_with_contours = image.copy()

    # Iterar sobre los contornos
    for contour in contours:
        # Obtener las coordenadas del cuadro delimitador
        x, y, w, h = cv2.boundingRect(contour)

        # Calcular el área del contorno
        area = cv2.contourArea(contour)

        # Verificar si el área es mayor o igual al porcentaje especificado del área total
        if (area / total_area) * 100 >= min_area_percentage:
            # Guardar la región de interés en la subcarpeta
            roi = image[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(subfolder_path, f"letter_{index}.png"), roi)

            # Dibujar el contorno en la imagen original
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

            index += 1

    # Guardar la imagen original con contornos verdes
    cv2.imwrite(os.path.join(subfolder_path, "original_image_with_contours.png"), image_with_contours)

# Ruta de la imagen de la placa de automóvil
image_path = "/Users/edisonlopez/Downloads/IA-1/II_Parcial/ExtraccionLetrasNumerosBarreraMorales/img20.jpg"  # Reemplaza con la ruta de tu imagen

# Porcentaje mínimo del área total para considerar un objeto
min_area_percentage_threshold = 2

detect_and_extract_letters(image_path, min_area_percentage_threshold)


if __name__ == "__main__":
    # Ruta de la imagen de la placa de automóvil
    image_path = "/Users/edisonlopez/Downloads/pl1.jpeg"  # Reemplaza con la ruta de tu imagen

    # Porcentaje mínimo del área total para considerar un objeto
    min_area_percentage_threshold = 2 

    detect_and_extract_letters(image_path, min_area_percentage_threshold)

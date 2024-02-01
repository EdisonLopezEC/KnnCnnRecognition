import cv2
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO
import os
import firebase_admin
from firebase_admin import credentials, firestore
import time
from tensorflow.keras.models import Sequential,load_model



cred = credentials.Certificate('/Users/edisonlopez/Downloads/respaldoFire/placas-2ca67-firebase-adminsdk-3999z-e850cbb840.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
# Clases de vehículos en el modelo YOLO
vehicle_classes = [2, 3, 5, 7]

# Mapeo de números a letras
number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
                    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
                    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}



# Parte 3: Predicción de la letra con el modelo KNN
class EuclideanDistanceKNN:
    def __init__(self, n_neighbors):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)
    
def detect_and_extract_letters(image, min_area_percentage, typical_aspect_ratio=0.5):
    
    blurred_image = cv2.GaussianBlur(image, (9, 9), 0)  # Aumentar el tamaño del kernel para un mayor desenfoque
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    total_area = image.shape[0] * image.shape[1]
    output_folder = "./extract"
    os.makedirs(output_folder, exist_ok=True)
    index = 1
    subfolder_path = os.path.join(output_folder, f"placa{index}")
    os.makedirs(subfolder_path, exist_ok=True)

    extracted_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h != 0 else 0  # Avoid division by zero

        # Add aspect ratio check and adjust x-coordinates
        if (area / total_area) * 100 >= min_area_percentage and 0.1 <= aspect_ratio <= 1.5:
            roi = image[y:y + h, max(0, x - 10):min(image.shape[1], x + w + 10)]
            image_name = os.path.join(subfolder_path, f"letter_{index}.png")
            cv2.imwrite(image_name, roi)
            extracted_images.append(image_name)
            index += 1

    return extracted_images

# def preprocess_image(input_path, output_path, vertical_padding):
#     input_image = Image.open(input_path).convert('L')
    
#     # Resize based on aspect ratio
#     input_image = input_image.filter(ImageFilter.EDGE_ENHANCE)

#     aspect_ratio = input_image.width / input_image.height
#     new_width = int(32 * aspect_ratio)
#     resized_image = input_image.resize((new_width, 32), Image.LANCZOS)
    
#     # Add horizontal padding
#     horizontal_padding = (32 - resized_image.width) // 2
#     padded_image = ImageOps.expand(resized_image, border=(horizontal_padding, vertical_padding), fill='white')
    
#     # Resize to final dimensions
#     final_image = padded_image.resize((32, 32), Image.LANCZOS)
    
#     # Apply additional smoothing
#     smoothed_image = final_image.filter(ImageFilter.SMOOTH_MORE)
    
#     # Save preprocessed image
#     smoothed_image.save(output_path)

def preprocess_image(input_path, output_path, vertical_padding):
    input_image = Image.open(input_path).convert('L')

    # Aumentar contraste
    enhancer = ImageEnhance.Contrast(input_image)
    contrasted_image = enhancer.enhance(2.0)

    final_image = contrasted_image

    # Redimensionar manteniendo la relación de aspecto
    aspect_ratio = final_image.width / final_image.height
    new_width = int(32 * aspect_ratio)
    resized_image = final_image.resize((new_width, 32), Image.LANCZOS)

    # Aplicar padding
    horizontal_padding = (32 - resized_image.width) // 2
    padded_image = ImageOps.expand(resized_image, border=(horizontal_padding, vertical_padding), fill='white')

    # Redimensionar al tamaño final
    final_image = padded_image.resize((32, 32), Image.LANCZOS)

    # Guardar la imagen resultante
    final_image.save(output_path)

def predict_letter(image_path, model):
    input_image = Image.open(image_path).convert('L')
    
    # Enhance contrast and adjust brightness
    enhanced_image = ImageEnhance.Contrast(input_image).enhance(1.5)
    enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(1.2)
    
    # Apply additional smoothing filter to reduce noise
    smoothed_image = enhanced_image.filter(ImageFilter.SMOOTH_MORE)
    
    # Resize with Lanczos interpolation
    resized_image = smoothed_image.resize((28, 28), Image.LANCZOS)
    
    # Normalize pixels and flatten the image
    normalized_pixels = np.array(resized_image) / 255.0
    flattened_pixels = normalized_pixels.flatten()
    
    # Perform prediction
    predicted_label = model.predict([flattened_pixels])[0]
    
    return predicted_label



def predict_letter_cnn(image_path, model):
    input_image = Image.open(image_path).convert('L')
    

    # Enhance contrast and adjust brightness
    enhanced_image = ImageEnhance.Contrast(input_image).enhance(1.5)
    enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(1.2)

    # Preprocesar la imagen
    resized_image = enhanced_image.resize((28, 28), Image.LANCZOS)
    normalized_pixels = np.array(resized_image) / 255.0
    reshaped_pixels = normalized_pixels.reshape(1, 28, 28, 1)  # Asegúrate de tener el formato correcto
    predicted_probs = model.predict(reshaped_pixels)[0]
    # Obtener la etiqueta predicha
    predicted_label = np.argmax(predicted_probs)
    
    return predicted_label


def process_license_plate(image_path, use_knn):

  

    # Mejorar la placa y guardar en la carpeta "placas_mejoradas"
    elements_images = detect_and_extract_letters(image_path, 1)
    plate_text = ""

    for i, element_img in enumerate(elements_images):
        preprocessed_img_path = element_img.replace(".png", "_processed.png")
        preprocess_image(element_img, preprocessed_img_path, 4)

        # Seleccionar el modelo adecuado según la opción
        if use_knn:
            if i < 3:
                model = knn_model_letras
            else:
                model = knn_model_numeros
            predicted_label = predict_letter(preprocessed_img_path, model)
            if i < 3:
                predicted_output = number_to_letter.get(predicted_label)
            else:
                predicted_output = str(predicted_label)
        else:
            if i < 3:
                model = cnn_model
            else:
                model = cnn_model_numeros
            predicted_label_cnn = predict_letter_cnn(preprocessed_img_path, model)

            if i < 3:
                predicted_output = number_to_letter.get(predicted_label_cnn, str(predicted_label_cnn))
            else:
                predicted_output = str(predicted_label_cnn)

        plate_text += predicted_output

    return plate_text


def send_to_firebase(plate_text, detection_time):
    if detection_time >= 5:
        # Envía la placa a Firestore
        doc_ref = db.collection('plates').document(plate_text)
        doc_ref.set({
            'count': plate_text,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

def detect_and_show_segmented_video(vehicle_model, license_plate_model):
    cap = cv2.VideoCapture(0)
    detection_frequency = 5
    recognized_plates = {}
    detection_start_time = 0

    try:

                   # Preguntar si se desea utilizar el modelo KNN o CNN por consola
        use_knn = input("¿Desea utilizar el modelo KNN caso contrario se usara CNN? (S/N): ")
        use_knn = use_knn.lower() == "s"
        if use_knn:
            print("Utilizando el modelo KNN")
        else:
            print("Utilizando el modelo CNN")

        while True:
            ret, frame = cap.read()


            if detection_frequency == 0:
                # Detect vehicles
                vehicle_detections = vehicle_model(frame)[0].boxes.data.tolist()

                # Detect license plates
                license_plate_detections = license_plate_model(frame)[0].boxes.data.tolist()

                for license_plate_detection in license_plate_detections:
                    x1, y1, x2, y2, score, class_id = license_plate_detection

                    # Extract and process license plate text
                    license_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    plate_text = process_license_plate(license_plate_region, use_knn)

                    # Add a rectangle around the license plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Update the dictionary of recognized plates
                    if plate_text not in recognized_plates:
                        recognized_plates[plate_text] = 1
                        detection_start_time = time.time()
                    else:
                        recognized_plates[plate_text] += 1

                    # Show the plate only if it has 6 digits and has been recognized 2 times
                    if len(plate_text) == 6 or len(plate_text) == 7 and recognized_plates[plate_text] >= 3:
                        # Increase font size
                        font_scale = 3
                        # Aumentar el grosor de la línea
                        thickness = 5

                        # Calculate detection time
                        detection_time = time.time() - detection_start_time

                        # Send to Firebase if detected for more than 5 seconds
                        send_to_firebase(plate_text, detection_time)

                        cv2.putText(frame, plate_text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

                detection_frequency = 1
            else:
                detection_frequency -= 1

            cv2.imshow('Real-time Segmentation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        vehicle_model.close()
        license_plate_model.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load YOLO models for vehicles and license plates
    coco_model = YOLO('yolov8n.pt')
    license_plate_model = YOLO('./models/license_plate_detector.pt')


    # Load KNN models for letters and numbers
    knn_model_letras_filename = "./knn_model_euclidean_letras.joblib"
    knn_model_letras = joblib.load(knn_model_letras_filename)

    # Cargar el modelo CNN para letras
    cnn_model = load_model('./cnn_model_letters.h5')

    # Cargar el modelo CNN para números
    cnn_model_numeros = load_model('./cnn_model_numbers.h5')


    knn_model_numeros_filename = "./knn_model_euclidean_numeros.joblib"
    knn_model_numeros = joblib.load(knn_model_numeros_filename)


    # Detect and show segmented video in real-time
    detect_and_show_segmented_video(coco_model, license_plate_model)

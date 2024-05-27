import cv2
import requests

# Cargar el clasificador de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# URL del endpoint de la API en tu servidor Django
api_url = 'http://192.168.100.7:8000/deteccion/detecciones/'

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

try:
    while True:
        # Capturar fotograma por fotograma
        ret, frame = cap.read()

        # Si el fotograma se capturó correctamente, ret es True
        if not ret:
            print("No se puede recibir el fotograma. Saliendo...")
            break

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Imprimir el número de caras detectadas en la consola
        print(f'Caras detectadas: {len(faces)}')

        # Enviar los datos de detección al servidor Django
        data = {'numero_de_caras': len(faces)}
        response = requests.post(api_url, data=data)

        # Mostrar el frame con las caras detectadas
        # cv2.imshow('Detección de Caras', frame)  # Comentado para evitar la creación de ventanas


finally:
    # Cuando todo esté hecho, liberar la captura y cerrar las ventanas
    cap.release()
    # cv2.destroyAllWindows()  # Esta línea no es necesaria con la nueva implementación

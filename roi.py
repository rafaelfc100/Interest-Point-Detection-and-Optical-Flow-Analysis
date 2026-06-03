import cv2

# Ruta de tu video
RUTA_VIDEO = r"C:\Users\rafae\OneDrive\Escritorio\Vision\ejercicio2\videoPieza.mov"

cap = cv2.VideoCapture(RUTA_VIDEO)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Leer el primer frame para seleccionar el ROI
ret, frame = cap.read()

if ret:
    # Instrucciones:
    # 1. Selecciona el área con el ratón.
    # 2. Presiona ENTER o ESPACIO para confirmar.
    # 3. Presiona 'c' para cancelar.
    print("\n--- SELECCIÓN DE ROI ---")
    print("1. Dibuja el rectángulo sobre la banda transportadora.")
    print("2. Presiona ENTER para confirmar.")
    
    roi = cv2.selectROI("Selecciona el ROI", frame, fromCenter=False, showCrosshair=True)
    
    # roi devuelve (x, y, w, h)
    print("\n========================================")
    print(f"Tus coordenadas de ROI son: {roi}")
    print(f"Copia esto en tu código: ROI = {roi}")
    print("========================================\n")
    
    cv2.destroyWindow("Selecciona el ROI")
else:
    print("No se pudo leer el frame.")

cap.release()
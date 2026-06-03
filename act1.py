import cv2
import numpy as np
import matplotlib.pyplot as plt

RUTA_VIDEO    = r"C:\Users\rafae\OneDrive\Escritorio\Vision\Tarea3\recordings\ejercicio_2.avi"
MAX_SEGUNDOS  = 60
TOP_PUNTOS    = 200
  


ROI = (0, 162, 638, 164)

"""
Harris: Es muy selectivo y se concentra en puntos de contraste extremo (esquinas).

FAST: Muestra el flujo de movimiento con estelas horizontales (ideal para seguimiento).

SIFT: Captura detalles de la textura de las cajas, creando una nube de puntos más densa
"""

def extraer_harris(gray, roi):
    x, y, w, h = roi
    roi_gray = np.float32(gray[y:y+h, x:x+w])
    dst = cv2.cornerHarris(roi_gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    umbral = 0.01 * dst.max() 
    coords = np.argwhere(dst > umbral)
    return [cv2.KeyPoint(float(c + x), float(r + y), 3, response=float(dst[r,c])) for r, c in coords]

cap = cv2.VideoCapture(RUTA_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
max_frames = int(MAX_SEGUNDOS * fps)

# detectores
fast = cv2.FastFeatureDetector_create(threshold=10)
sift = cv2.SIFT_create()
acum = {"Harris": [], "FAST": [], "SIFT": []}

f_idx = 0
while f_idx < max_frames:
    ret, frame = cap.read()
    if not ret: break
    
    if f_idx % 2 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] = 255

        pts_harris = extraer_harris(gray, ROI)
        acum["Harris"].extend(pts_harris)
        
        pts_fast = fast.detect(gray, mask=mask)
        acum["FAST"].extend(pts_fast)
        
        pts_sift, _ = sift.detectAndCompute(gray, mask=mask)
        acum["SIFT"].extend(pts_sift)

        frame_vis = frame.copy()
        

        cv2.rectangle(frame_vis, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (255, 255, 255), 1)
        
        # fast en verde
        cv2.drawKeypoints(frame_vis, pts_harris, frame_vis, color=(0, 255, 0))
        cv2.imshow("Visualizacion de Procesamiento", frame_vis)

    f_idx += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Procesamiento terminado.")
for nombre, puntos in acum.items():
    mejores = sorted(puntos, key=lambda x: -x.response)[:TOP_PUNTOS]
    pts = np.array([kp.pt for kp in mejores])

    plt.figure(figsize=(10, 6))
    plt.gcf().patch.set_facecolor('white')
    
    if len(pts) > 0:
        color = 'red' if nombre == "Harris" else 'green' if nombre == "FAST" else 'purple'
        plt.scatter(pts[:, 0], pts[:, 1], c=color, s=15, alpha=0.6, edgecolors='none')

    plt.gca().add_patch(plt.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], 
                                     fill=False, edgecolor='gray', linestyle='--', alpha=0.5))
    
    plt.title(f"{nombre}\n200 puntos", 
              fontsize=14, fontweight='bold')
    plt.xlim(0, 640); plt.ylim(480, 0) # Ajustar a resolución del video
    #plt.xlabel("Ancho (px)"); plt.ylabel("Alto (px)")
    plt.grid(True, linestyle=':', alpha=0.3)
    
    nombre_archivo = f"{nombre}.png"
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()
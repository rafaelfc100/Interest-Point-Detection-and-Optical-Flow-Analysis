import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Configuración de matplotlib para fondo blanco y estilo tradicional
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

ROI = (211, 1, 333, 477)

VELOCITY_THRESHOLD = 2.5   
ALPHA = 0.5
CLUSTER_DISTANCE = 40
MIN_POINTS_CLUSTER = 3
MAX_OBJECTS = 3

COLOR_TEXT_MAIN = (255, 0, 255)
COLOR_TEXT_OBJ = (0, 255, 255)
COLOR_POINTS = (0, 140, 255)
COLOR_ARROWS = (0, 180, 255)
COLOR_BBOX_PREV = (0, 0, 255)
COLOR_BBOX_CURR = (255, 0, 0)
COLOR_ROI = (255, 255, 0)

MAX_MASK_VECTORS = 200
mask_vectors = []

# Variables para almacenar datos de flujo para el plot
flow_vectors = []  # Almacenar vectores de flujo actuales

def cluster_points_kmeans(points, k=MAX_OBJECTS):
    if len(points) < 2:
        return [points] if len(points) > 0 else []
    
    points_f = points.astype(np.float32)
    best_clusters = None
    best_score = np.inf
    prev_inertia = None
    
    for ki in range(1, min(k + 1, len(points) + 1)):
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        ret, labels, centers = cv2.kmeans(points_f, ki, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        inertia = 0.0
        for ci in range(ki):
            cluster_pts = points_f[labels.ravel() == ci]
            if len(cluster_pts) > 0:
                inertia += np.sum((cluster_pts - centers[ci]) ** 2)
        
        if prev_inertia is None or prev_inertia == 0:
            prev_inertia = inertia if inertia > 0 else 1.0
            best_clusters = ki
            best_score = inertia
        else:
            drop = (prev_inertia - inertia) / prev_inertia
            if drop > 0.30:
                best_clusters = ki
                best_score = inertia
            prev_inertia = inertia
    
    final_k = best_clusters if best_clusters else 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    _, labels, _ = cv2.kmeans(points_f, final_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    clusters = []
    for ci in range(final_k):
        mask = labels.ravel() == ci
        cluster = points[mask]
        if len(cluster) >= MIN_POINTS_CLUSTER:
            clusters.append(cluster)
    
    clusters.sort(key=len, reverse=True)
    return clusters

def refine_cluster(cluster, max_radius=80):
    cluster = np.array(cluster)
    centroid = np.mean(cluster, axis=0)
    dists = np.linalg.norm(cluster - centroid, axis=1)
    return cluster[dists < max_radius]

def get_bounding_box(points, padding=15):
    if len(points) == 0:
        return None
    pts = np.array(points)
    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))
    return (x_min - padding, y_min - padding,
            (x_max - x_min) + 2 * padding,
            (y_max - y_min) + 2 * padding)

def smooth_bbox(prev, curr, alpha=ALPHA):
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
        int(alpha * prev[2] + (1 - alpha) * curr[2]),
        int(alpha * prev[3] + (1 - alpha) * curr[3]),
    )

def match_and_smooth_bboxes(prev_list, curr_list, alpha=ALPHA):
    if not prev_list:
        return curr_list
    
    smoothed = []
    used_prev = set()
    
    for curr in curr_list:
        cx = curr[0] + curr[2] / 2
        cy = curr[1] + curr[3] / 2
        best_idx = None
        best_dist = np.inf
        
        for i, prev in enumerate(prev_list):
            if i in used_prev:
                continue
            px = prev[0] + prev[2] / 2
            py = prev[1] + prev[3] / 2
            d = np.hypot(cx - px, cy - py)
            if d < best_dist:
                best_dist = d
                best_idx = i
        
        if best_idx is not None and best_dist < 200:
            used_prev.add(best_idx)
            smoothed.append(smooth_bbox(prev_list[best_idx], curr, alpha))
        else:
            smoothed.append(curr)
    
    return smoothed

def draw_hud(output, frame_count, max_frames, velocity_threshold, n_objects, n_moving):
    lines = [
        (f"Frame {frame_count}/{max_frames}", COLOR_TEXT_MAIN, (10, 25)),
        (f"Umbral velocidad: {velocity_threshold:.1f} px", COLOR_TEXT_MAIN, (10, 75)),
    ]
    
    overlay = output.copy()
    cv2.rectangle(overlay, (5, 5), (320, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    
    for text, color, pos in lines:
        cv2.putText(output, text, (pos[0]+1, pos[1]+1),
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(output, text, pos,
                    font, font_scale, color, thickness)

def plot_optical_flow(vectors, frame_shape):
    """Crea un plot tradicional con fondo blanco del flujo óptico vectorial"""
    plt.clf()
    
    # Crear figura con tamaño adecuado
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    
    # Configurar el plot
    ax = plt.gca()
    
    if len(vectors) > 0:
        # Separar puntos y vectores
        points = np.array([v[0] for v in vectors])
        flows = np.array([v[1] - v[0] for v in vectors])
        
        # Calcular magnitudes para el coloreado
        magnitudes = np.sqrt(flows[:, 0]**2 + flows[:, 1]**2)
        
        # Normalizar vectores para mejor visualización (escalar para que se vean bien)
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 1
        if max_mag > 0:
            scale = min(30, 100 / max_mag)  # Escala adaptativa
            flows_scaled = flows * scale
        else:
            flows_scaled = flows
        
        # Crear el quiver plot con colores según magnitud
        quiver = ax.quiver(points[:, 0], points[:, 1], 
                          flows_scaled[:, 0], flows_scaled[:, 1],
                          magnitudes, 
                          cmap='viridis', 
                          scale=1, 
                          scale_units='xy',
                          angles='xy',
                          width=0.003,
                          headwidth=3,
                          headlength=4)
        
        # Añadir barra de color
        cbar = plt.colorbar(quiver)
        cbar.set_label('Magnitud del flujo (px/frame)', fontsize=10, color='black')
        cbar.ax.yaxis.label.set_color('black')
        cbar.ax.tick_params(colors='black')
        
        # Añadir puntos en los orígenes
        ax.scatter(points[:, 0], points[:, 1], c='red', s=20, alpha=0.6, 
                  edgecolors='black', linewidth=0.5, label='Puntos de interés')
    
    # Configurar el plot
    ax.set_xlim(0, frame_shape[1])
    ax.set_ylim(frame_shape[0], 0)  # Invertir Y para que coincida con imagen
    ax.set_xlabel('X (pixeles)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (pixeles)', fontsize=12, fontweight='bold')
    ax.set_title('Flujo Óptico Vectorial', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    
    # Añadir información estadística
    if len(vectors) > 0:
        avg_magnitude = np.mean(magnitudes)
        max_magnitude = np.max(magnitudes)
        info_text = f'Vectores: {len(vectors)} | Vel. media: {avg_magnitude:.2f} | Vel. max: {max_magnitude:.2f}'
        plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# Inicializar figura de matplotlib (fondo blanco)
plt.ion()  # Modo interactivo
fig = plt.figure(figsize=(10, 8), facecolor='white')
fig.suptitle('Análisis de Flujo de Movimiento', fontsize=16, fontweight='bold', y=0.98)

# Inicializar captura de video
cap = cv2.VideoCapture(r"C:\Users\rafae\OneDrive\Escritorio\Vision\ejercicio2\videoPieza.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
max_frames = int(fps * 60)
print(f"FPS: {fps:.2f} | Procesando {max_frames} frames (1 min)")

ret, old_frame = cap.read()
if not ret:
    print("Error al leer el video")
    cap.release()
    exit()

x, y, w, h = ROI
old_frame_roi = old_frame[y:y+h, x:x+w]
old_gray = cv2.cvtColor(old_frame_roi, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
if p0 is not None:
    p0 = p0.reshape(-1, 2)
    p0[:, 0] += x
    p0[:, 1] += y

frame_count = 0
bboxes_prev = []
bboxes_current = []
frame_shape = old_frame.shape

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame_roi = frame[y:y+h, x:x+w]
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    
    output = frame.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), COLOR_ROI, 2)
    
    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        if p0 is not None:
            p0 = p0.reshape(-1, 2)
            p0[:, 0] += x
            p0[:, 1] += y
        else:
            old_gray = frame_gray.copy()
            continue
    
    # Flujo óptico
    p0_roi = p0.copy()
    p0_roi[:, 0] -= x
    p0_roi[:, 1] -= y
    p0_lk = p0_roi.reshape(-1, 1, 2).astype(np.float32)
    
    p1_lk, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_lk, None, **lk_params)
    
    if p1_lk is None or st is None:
        old_gray = frame_gray.copy()
        continue
    
    p1_roi_valid = p1_lk[st == 1]
    p0_roi_valid = p0_lk[st == 1]
    
    p1_valid = p1_roi_valid.reshape(-1, 2)
    p0_valid = p0_roi_valid.reshape(-1, 2)
    
    p1_valid[:, 0] += x
    p1_valid[:, 1] += y
    p0_valid[:, 0] += x
    p0_valid[:, 1] += y
    
    vx = p1_valid[:, 0] - p0_valid[:, 0]
    vy = p1_valid[:, 1] - p0_valid[:, 1]
    velocities = np.sqrt(vx**2 + vy**2)
    
    moving_mask = (velocities > VELOCITY_THRESHOLD) & (velocities < 50)
    moving_points_current = p1_valid[moving_mask]
    moving_points_prev = p0_valid[moving_mask]
    
    # Actualizar plot de flujo óptico cada cierto número de frames
    if frame_count % 3 == 0 and len(moving_points_current) > 0:
        vectors = list(zip(moving_points_prev, moving_points_current))
        plot_optical_flow(vectors, frame_shape)
    
    # Dibujar vectores de flujo en la ventana de video
    for new, old_pt in zip(moving_points_current, moving_points_prev):
        cv2.arrowedLine(output,
                        (int(old_pt[0]), int(old_pt[1])),
                        (int(new[0]), int(new[1])),
                        COLOR_ARROWS, 2, tipLength=0.4)
        cv2.circle(output, (int(new[0]), int(new[1])), 3, COLOR_POINTS, -1)
    
    # Detectar objetos
    bboxes_current = []
    if len(moving_points_current) >= MIN_POINTS_CLUSTER:
        clusters = cluster_points_kmeans(moving_points_current)
        
        for cluster in clusters:
            refined = refine_cluster(cluster)
            if len(refined) < MIN_POINTS_CLUSTER:
                continue
            bbox = get_bounding_box(refined)
            if bbox and bbox[2] > 10 and bbox[3] > 10:
                bboxes_current.append(bbox)
    
    bboxes_current = match_and_smooth_bboxes(bboxes_prev, bboxes_current)
    
    # Dibujar bounding boxes
    for i, (bx, by, bw, bh) in enumerate(bboxes_prev):
        cv2.rectangle(output, (bx, by), (bx + bw, by + bh), COLOR_BBOX_PREV, 3)
        cv2.putText(output, "Anterior", (bx + 2, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BBOX_PREV, 2)
    
    for i, (bx, by, bw, bh) in enumerate(bboxes_current):
        cv2.rectangle(output, (bx, by), (bx + bw, by + bh), COLOR_BBOX_CURR, 3)
        cv2.putText(output, "Actual", (bx + 2, by + bh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BBOX_CURR, 2)
    
    bboxes_prev = bboxes_current.copy()
    p0 = p1_valid.copy()
    old_gray = frame_gray.copy()
    draw_hud(output, frame_count, max_frames,
             VELOCITY_THRESHOLD, len(bboxes_current), len(moving_points_current))
    
    cv2.imshow("Flujo Optico + Tracking", output)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('+'), ord('=')):
        VELOCITY_THRESHOLD = min(VELOCITY_THRESHOLD + 0.5, 20.0)
        print(f"Umbral: {VELOCITY_THRESHOLD}")
    elif key in (ord('-'), ord('_')):
        VELOCITY_THRESHOLD = max(VELOCITY_THRESHOLD - 0.5, 0.5)
        print(f"Umbral: {VELOCITY_THRESHOLD}")

# Mostrar gráfico final
plt.ioff()
plt.show(block=True)

print("Procesamiento completado")
cap.release()
cv2.destroyAllWindows()
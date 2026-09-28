"""QCar: avance en LINEA RECTA + eventos viales con YOLOv5n (COCO) + ROI.

Uso (en el QCar, en la misma carpeta que camera.py):
  python3 qcar_recta_eventos.py --probe    # imprime sensores, NO mueve el carro
  python3 qcar_recta_eventos.py            # solo percepcion (NO mueve el carro)
  python3 qcar_recta_eventos.py --drive    # recta + eventos (EL CARRO SE MUEVE)

Hilo principal = percepcion (camara -> YOLO -> ROI -> eventos).
Hilo de control = mantener rumbo recto con giroscopio + supervisor de velocidad (50 Hz).

Eventos activos:
  STOP_APPROACH -> letrero visto a lo lejos: mitad de velocidad (llega mas controlado)
  STOP_SIGN     -> letrero cerca: paro temporal 3 s, luego sigue (cooldown para no re-disparar)
  PERSON        -> persona en cualquier parte de la imagen: paro mientras siga visible
  GIRAFFE_TURN  -> jirafa (imagen): giro de 90 grados a velocidad crucero y luego sigue recto
                   (YOLO + validacion por color amarillo + detector de respaldo por color)
Desactivados (comentados en make_events): RED_LIGHT, YELLOW_LIGHT.

Sin limite de distancia ni de tiempo: el carro se detiene con Ctrl+C en la terminal
(o tecla q / Esc en la ventana de video).

Deteccion del STOP (para verlo antes):
  1. YOLO corre solo sobre el recorte del ROI -> el letrero se ve ~2x mas grande.
  2. CLAHE mejora contraste en iluminacion pobre (solo la entrada de YOLO).
  3. Validacion por color rojo: una deteccion YOLO de baja confianza se acepta si
     su recuadro es suficientemente rojo.
  4. Detector de respaldo por color + forma (mancha roja casi octagonal), por si
     YOLO no lo ve a distancia.
"""

import argparse
import csv
import math
import os
import threading
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort

from camera import CameraProcessor

try:
    from pal.products.qcar import QCar
except ImportError:
    QCar = None

# ------------------------------------------------------------------ CONFIG
MODEL_PATH = "/home/nvidia/Assesment_qcar_irs/models/yolov5n.onnx"
CAMERA_ID = "3"
IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_RATE = 1640, 820, 30
PRE_CONF = 0.25          # umbral minimo de YOLO (el filtro fino es STOP_CONF_*)
NMS_THRESHOLD = 0.45

# --- Deteccion del STOP
ENHANCE = True           # CLAHE sobre la entrada de YOLO
STOP_CONF_HIGH = 0.50    # YOLO >= esto: se acepta directo
STOP_CONF_LOW = 0.25     # YOLO entre LOW y HIGH: se acepta si el recuadro es rojo
STOP_RED_MIN = 0.20      # fraccion minima de pixeles rojos en el recuadro
ENABLE_COLOR_DETECTOR = True
COLOR_MIN_AREA = 500     # px (imagen completa) de la mancha roja minima
RED_S_MIN, RED_V_MIN = 100, 60   # saturacion / brillo minimos para "rojo"

# --- Jirafa (COCO "giraffe") -> giro de 90 grados
GIRAFFE_CONF_HIGH = 0.45     # YOLO >= esto: se acepta directo
GIRAFFE_CONF_LOW = 0.25      # YOLO entre LOW y HIGH: se acepta si el recuadro es amarillo
GIRAFFE_YELLOW_MIN = 0.15    # fraccion minima de pixeles amarillos en el recuadro
GIRAFFE_MIN_H = 0.05         # alto minimo del recuadro / alto de imagen
GIRAFFE_COOLDOWN = 8.0       # s sin volver a disparar un giro (misma jirafa)
ENABLE_YELLOW_DETECTOR = True    # respaldo sin YOLO: mancha amarilla con manchas cafes
YELLOW_MIN_AREA = 2000       # px (imagen completa) de la mancha minima
YELLOW_H = (15, 35)          # rango de tono (OpenCV 0-179) del amarillo
YELLOW_S_MIN, YELLOW_V_MIN = 80, 110
BROWN_H = (3, 22)            # tono de las manchas cafes de la jirafa
BROWN_S_MIN, BROWN_V_RANGE = 70, (30, 110)
TURN_DIR = 1             # +1 = izquierda, -1 = derecha (depende de STEER_SIGN calibrado)
TURN_ANGLE = math.radians(90)
TURN_TOL = math.radians(5)   # error de rumbo para dar el giro por terminado

VEHICLE_IO_RATE = 200
CONTROL_RATE = 50        # Hz
STEER_MAX = 0.5          # rad
KP_HEADING = 1.0         # rad de direccion por rad de desvio de rumbo
STEER_SIGN = 1.0         # cambia a -1.0 si el carro corrige hacia el lado equivocado
THROTTLE_CRUISE = 0.05
THROTTLE_MIN = 0.035     # debajo de esto el motor no vence la friccion (calibrar)
ACCEL_RATE = 0.10        # throttle/s al arrancar (suave)
BRAKE_RATE = 1.00        # throttle/s al frenar (casi inmediato)
STALE_S = 0.7            # percepcion sin actualizar -> paro de seguridad
MAX_RUN_S = None         # s de tope de la corrida; None = sin limite (parar con Ctrl+C)

# Clases COCO
PERSON, TRAFFIC_LIGHT, STOP_SIGN, GIRAFFE = 0, 9, 11, 23

LOG_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ EVENTOS
class RoadEvent:
    """Evento vial: clase COCO + ROI + confianza + persistencia -> factor de velocidad.

    action = "timed_stop": paro `duration` s y luego `cooldown` s de gracia.
    action = "hold":       aplica `factor` mientras el objeto siga en el ROI.
    action = "trigger":    no cambia la velocidad; marca `fired` un frame (p. ej. pedir
                           un giro) y espera `cooldown` s antes de poder volver a disparar.
    ROI en fracciones de la imagen (x0, y0, x1, y1); min_h = alto minimo del recuadro
    relativo a la imagen (proxy de cercania); color = solo para semaforos.
    """

    def __init__(self, name, cls_id, roi, min_conf, persist, action, min_h=0.0,
                 factor=0.0, duration=3.0, cooldown=6.0, clear_frames=5, color=None):
        self.name, self.cls_id, self.roi = name, cls_id, roi
        self.min_conf, self.persist, self.action = min_conf, persist, action
        self.min_h, self.factor, self.color = min_h, factor, color
        self.duration, self.cooldown, self.clear_frames = duration, cooldown, clear_frames
        self.active, self.count, self.clear = False, 0, 0
        self.t_end, self.t_cool = 0.0, 0.0
        self.fired = False

    def _set(self, state, now, log):
        self.active = state
        log.append((round(now, 3), self.name, "ON" if state else "OFF"))
        print("[%.2f s] %s %s" % (now, self.name, "ON" if state else "OFF"))

    def update(self, dets, shape, now, log):
        H, W = shape[:2]
        x0, y0, x1, y1 = self.roi
        hit = False
        for d in dets:
            if d["cls"] != self.cls_id or d["conf"] < self.min_conf:
                continue
            if self.color and d.get("color") != self.color:
                continue
            x, y, w, h = d["box"]
            cx, cy = x + w / 2, y + h / 2
            if x0 * W <= cx <= x1 * W and y0 * H <= cy <= y1 * H and h / H >= self.min_h:
                hit = True
                d["used"] = True
        if self.action == "trigger":
            self.fired = False
            self.count = self.count + 1 if hit else 0
            if self.count >= self.persist and now >= self.t_cool:
                self.fired, self.count = True, 0
                self.t_cool = now + self.cooldown
                log.append((round(now, 3), self.name, "FIRE"))
                print("[%.2f s] %s FIRE" % (now, self.name))
            return 1.0
        elif self.action == "timed_stop":
            if self.active:
                if now >= self.t_end:
                    self._set(False, now, log)
                    self.t_cool = now + self.cooldown
            else:
                self.count = self.count + 1 if hit else 0
                if self.count >= self.persist and now >= self.t_cool:
                    self._set(True, now, log)
                    self.t_end = now + self.duration
                    self.count = 0
        else:
            if hit:
                self.count, self.clear = self.count + 1, 0
            else:
                self.count, self.clear = 0, self.clear + 1
            if not self.active and self.count >= self.persist:
                self._set(True, now, log)
            elif self.active and self.clear >= self.clear_frames:
                self._set(False, now, log)
        return self.factor if self.active else 1.0


def make_events():
    stop_roi = (0.00, 0.00, 1.00, 1.00)
    # light_roi = (0.30, 0.00, 1.00, 0.60)
    return [
        # min_h = alto del letrero / alto de imagen: 0.04 ~ lejos, 0.10 ~ cerca (calibrar)
        RoadEvent("STOP_APPROACH", STOP_SIGN, stop_roi, STOP_CONF_LOW, 2, "hold",
                  min_h=0.04, factor=0.5, clear_frames=8),
        RoadEvent("STOP_SIGN", STOP_SIGN, stop_roi, STOP_CONF_LOW, 3, "timed_stop",
                  min_h=0.10, duration=3.0, cooldown=6.0),
        # Persona en cualquier parte de la imagen: paro mientras siga visible (reanuda al salir)
        RoadEvent("PERSON", PERSON, (0.00, 0.00, 1.00, 1.00), 0.50, 2, "hold",
                  min_h=0.25, factor=0.0, clear_frames=5),
        # Jirafa en cualquier parte de la imagen: pide UN giro de 90 grados
        RoadEvent("GIRAFFE_TURN", GIRAFFE, (0.00, 0.00, 1.00, 1.00), GIRAFFE_CONF_LOW, 3,
                  "trigger", min_h=GIRAFFE_MIN_H, cooldown=GIRAFFE_COOLDOWN),
        # --- Desactivados por ahora ---
        # RoadEvent("RED_LIGHT", TRAFFIC_LIGHT, light_roi, 0.40, 3, "hold",
        #           min_h=0.06, factor=0.0, clear_frames=8, color="red"),
        # RoadEvent("YELLOW_LIGHT", TRAFFIC_LIGHT, light_roi, 0.40, 3, "hold",
        #           min_h=0.06, factor=0.5, clear_frames=5, color="yellow"),
    ]


# ------------------------------------------------------------------ COMPARTIDO
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.factor, self.stamp, self.active = 1.0, 0.0, []
        self.ready = False
        self.stop = threading.Event()
        self.telemetry = {}
        self.turn_requests = 0

    def request_turn(self):
        with self.lock:
            self.turn_requests += 1

    def get_turn_requests(self):
        with self.lock:
            return self.turn_requests

    def set_events(self, factor, active):
        with self.lock:
            self.factor, self.stamp, self.active = factor, time.time(), active

    def get_events(self):
        with self.lock:
            return self.factor, time.time() - self.stamp, list(self.active)


# ------------------------------------------------------------------ PERCEPCION
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shp = inp.shape
    in_h = int(shp[2]) if isinstance(shp[2], int) else 640
    in_w = int(shp[3]) if isinstance(shp[3], int) else 640
    meta = (inp.name, in_w, in_h, [o.name for o in session.get_outputs()])
    return session, meta


def nms_indices(boxes, scores, score_thr, iou_thr):
    idxs = np.nonzero(scores >= score_thr)[0]
    if idxs.size == 0:
        return []
    boxes, scores = boxes[idxs], scores[idxs]
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(int(idxs[i]))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_thr]
    return kept


def detect(session, meta, frame, class_ids):
    in_name, in_w, in_h, out_names = meta
    H, W = frame.shape[:2]
    r = min(in_w / W, in_h / H)
    nw, nh = int(round(W * r)), int(round(H * r))
    top, left = (in_h - nh) // 2, (in_w - nw) // 2
    canvas = np.full((in_h, in_w, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    out = np.asarray(session.run(out_names, {in_name: tensor[None]})[0])
    rows = out.reshape(-1, out.shape[-1])
    rows = rows[rows[:, 4] >= PRE_CONF]
    if rows.size == 0:
        return []
    cls_scores = rows[:, 5:]
    cid = cls_scores.argmax(1)
    conf = rows[:, 4] * cls_scores[np.arange(len(rows)), cid]
    keep = (conf >= PRE_CONF) & np.isin(cid, class_ids)
    rows, cid, conf = rows[keep], cid[keep], conf[keep]
    if len(rows) == 0:
        return []
    x = (rows[:, 0] - rows[:, 2] / 2 - left) / r
    y = (rows[:, 1] - rows[:, 3] / 2 - top) / r
    boxes = np.stack([x, y, rows[:, 2] / r, rows[:, 3] / r], 1)
    # NMS por clase: desplazar cada clase evita que una persona suprima a un STOP encimado
    shifted = boxes.copy()
    shifted[:, :2] += cid[:, None] * 10000.0
    idx = nms_indices(shifted, conf, PRE_CONF, NMS_THRESHOLD)
    return [dict(cls=int(cid[i]), conf=float(conf[i]),
                 box=tuple(float(v) for v in boxes[i])) for i in idx]


def box_crop(frame, box):
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    return frame[y0:y1, x0:x1]


def union_roi(events):
    return (min(ev.roi[0] for ev in events), min(ev.roi[1] for ev in events),
            max(ev.roi[2] for ev in events), max(ev.roi[3] for ev in events))


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(bgr):
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((h <= 10) | (h >= 165)) & (s >= RED_S_MIN) & (v >= RED_V_MIN)


def red_fraction(frame, box):
    crop = box_crop(frame, box)
    if crop.size == 0:
        return 0.0
    return float(np.mean(red_mask(crop)))


def hsv_range(bgr, h_rng, s_min, v_rng):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((h >= h_rng[0]) & (h <= h_rng[1]) & (s >= s_min)
            & (v >= v_rng[0]) & (v <= v_rng[1]))


def yellow_mask(bgr):
    return hsv_range(bgr, YELLOW_H, YELLOW_S_MIN, (YELLOW_V_MIN, 255))


def brown_mask(bgr):
    return hsv_range(bgr, BROWN_H, BROWN_S_MIN, BROWN_V_RANGE)


def yellow_fraction(frame, box):
    crop = box_crop(frame, box)
    if crop.size == 0:
        return 0.0
    return float(np.mean(yellow_mask(crop)))


def giraffe_ok(d):
    if d["conf"] >= GIRAFFE_CONF_HIGH:
        return True
    return d["conf"] >= GIRAFFE_CONF_LOW and d.get("yellow", 0.0) >= GIRAFFE_YELLOW_MIN


def yellow_giraffes(crop, ox, oy):
    """Respaldo sin YOLO: zona amarilla con manchas cafes (patron de jirafa)."""
    ym, bm = yellow_mask(crop), brown_mask(crop)
    m = (ym | bm).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < YELLOW_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        yellow = float(np.mean(ym[y:y + h, x:x + w]))
        brown = float(np.mean(bm[y:y + h, x:x + w]))
        # Amarillo dominante + algo de cafe: descarta superficies de un solo tono
        if yellow < 0.25 or not 0.05 <= brown <= 0.45:
            continue
        out.append(dict(cls=GIRAFFE, conf=0.50, box=(x + ox, y + oy, w, h), src="color",
                        yellow=yellow))
    return out


def stop_ok(d):
    if d["conf"] >= STOP_CONF_HIGH:
        return True
    return d["conf"] >= STOP_CONF_LOW and d.get("red", 0.0) >= STOP_RED_MIN


def red_octagons(crop, ox, oy):
    """Respaldo sin YOLO: manchas rojas compactas, casi cuadradas y de ~8 lados."""
    m = red_mask(crop).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < COLOR_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not 0.7 <= w / float(h) <= 1.4:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or area / hull < 0.85:
            continue
        if np.count_nonzero(m[y:y + h, x:x + w]) / float(w * h) < 0.45:
            continue
        sides = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))
        if not 6 <= sides <= 10:
            continue
        out.append(dict(cls=STOP_SIGN, conf=0.50, box=(x + ox, y + oy, w, h), src="color"))
    return out


def inside(d, boxes):
    x, y, w, h = d["box"]
    cx, cy = x + w / 2, y + h / 2
    return any(bx <= cx <= bx + bw and by <= cy <= by + bh for bx, by, bw, bh in boxes)


def light_color(frame, box):
    """COCO solo dice 'traffic light', no el color: se clasifica por HSV dentro del recuadro."""
    crop = box_crop(frame, box)
    if crop.shape[0] < 6 or crop.shape[1] < 3:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    lit = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 130)
    counts = {
        "red": np.count_nonzero(lit & ((hue <= 10) | (hue >= 160))),
        "yellow": np.count_nonzero(lit & (hue >= 15) & (hue <= 35)),
        "green": np.count_nonzero(lit & (hue >= 45) & (hue <= 95)),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] >= 0.02 * hue.size else "unknown"


def draw(frame, dets, events, factor, tel, fps, show_roi=True):
    H, W = frame.shape[:2]
    if show_roi:
        groups = {}
        for ev in events:
            groups.setdefault(ev.roi, []).append(ev)
        for i, ((x0, y0, x1, y1), evs) in enumerate(groups.items()):
            active = any(ev.active for ev in evs)
            col = (0, 0, 255) if active else (255, 0, 0)
            p0 = (int(x0 * W), int(y0 * H))
            p1 = (int(x1 * W) - 2, int(y1 * H) - 2)
            cv2.rectangle(frame, p0, p1, col, 3 if active else 1)
            cv2.putText(frame, "ROI " + " / ".join(ev.name for ev in evs),
                        (p0[0] + 6, p0[1] + 22 + 22 * (i % 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    names = {PERSON: "person", TRAFFIC_LIGHT: "light", STOP_SIGN: "stop", GIRAFFE: "jirafa"}
    for d in dets:
        x, y, w, h = [int(v) for v in d["box"]]
        col = (0, 255, 0) if d.get("used") else (0, 165, 255)
        if d.get("src") == "color":
            label = "%s(color) h=%.2f" % (names.get(d["cls"], d["cls"]), h / float(H))
        else:
            label = "%s %.2f" % (names.get(d["cls"], d["cls"]), d["conf"])
        if "red" in d:
            label += " rojo=%.2f" % d["red"]
        if "yellow" in d:
            label += " amarillo=%.2f" % d["yellow"]
        if "color" in d:
            label += " " + d["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        cv2.putText(frame, label, (max(0, x), max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    lines = ["Factor vel: %.2f  FPS: %.1f" % (factor, fps)]
    if tel:
        lines.append("Estado: %s  v=%.2f m/s  thr=%.3f" % (tel["state"], tel["v"], tel["thr"]))
        lines.append("dist=%.2f m  rumbo=%.0f ref=%.0f deg  steer=%.2f" % (
            tel["dist"], math.degrees(tel["th"]), math.degrees(tel["ref"]), tel["steer"]))
    y_top = H - 20 - 32 * len(lines)
    cv2.rectangle(frame, (0, y_top), (620, H), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, y_top + 28 + 32 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run_perception(shared, events, display, event_log):
    session, meta = load_model()
    print("ONNX Runtime CPU | entrada:", meta[0], meta[1], "x", meta[2])
    class_ids = sorted({ev.cls_id for ev in events})
    crop_roi = union_roi(events)
    session.run(meta[3], {meta[0]: np.zeros((1, 3, meta[2], meta[1]), np.float32)})
    camera = CameraProcessor(CAMERA_ID, IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_RATE)
    shared.ready = True
    t0, fps, tprev = time.perf_counter(), 0.0, time.perf_counter()
    show_roi = True
    try:
        while not shared.stop.is_set():
            frame = camera.take_frame()
            if frame is None or frame.size == 0:
                continue  # sin frame no se actualiza el sello -> el control frena solo
            now = time.perf_counter() - t0
            H, W = frame.shape[:2]
            ox, oy = int(crop_roi[0] * W), int(crop_roi[1] * H)
            crop = frame[oy:int(crop_roi[3] * H), ox:int(crop_roi[2] * W)]
            dets = detect(session, meta, enhance(crop) if ENHANCE else crop, class_ids)
            for d in dets:
                x, y, w, h = d["box"]
                d["box"] = (x + ox, y + oy, w, h)
                d["src"] = "yolo"
                if d["cls"] == STOP_SIGN:
                    d["red"] = red_fraction(frame, d["box"])
                if d["cls"] == GIRAFFE:
                    d["yellow"] = yellow_fraction(frame, d["box"])
                # if d["cls"] == TRAFFIC_LIGHT:
                #     d["color"] = light_color(frame, d["box"])
            dets = [d for d in dets if d["cls"] != STOP_SIGN or stop_ok(d)]
            dets = [d for d in dets if d["cls"] != GIRAFFE or giraffe_ok(d)]
            if ENABLE_COLOR_DETECTOR:
                yolo_stops = [d["box"] for d in dets if d["cls"] == STOP_SIGN]
                dets += [c for c in red_octagons(crop, ox, oy) if not inside(c, yolo_stops)]
            if ENABLE_YELLOW_DETECTOR:
                yolo_giraffes = [d["box"] for d in dets if d["cls"] == GIRAFFE]
                dets += [c for c in yellow_giraffes(crop, ox, oy)
                         if not inside(c, yolo_giraffes)]
            factors = [ev.update(dets, frame.shape, now, event_log) for ev in events]
            # Mientras un paro temporal esta activo o en cooldown, se ignoran los demas
            # eventos de la misma clase (p. ej. STOP_APPROACH) para que el carro reanude.
            muted = {ev.cls_id for ev in events
                     if ev.action == "timed_stop" and (ev.active or now < ev.t_cool)}
            live = [(ev, f) for ev, f in zip(events, factors)
                    if ev.action == "timed_stop" or ev.cls_id not in muted]
            factor = min([f for _, f in live] + [1.0])
            shared.set_events(factor, [ev.name for ev, _ in live if ev.active])
            if any(ev.fired for ev in events):
                shared.request_turn()
            tn = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(tn - tprev, 1e-6)
            tprev = tn
            if display:
                draw(frame, dets, events, factor, shared.telemetry, fps, show_roi)
                cv2.imshow("QCar recta + eventos", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    show_roi = not show_roi
    finally:
        shared.stop.set()
        camera.end_camera()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ CONTROL
def open_vehicle():
    if QCar is None:
        raise RuntimeError("API Quanser QCar no disponible (revisa PYTHONPATH=$QPATH)")
    car = QCar(readMode=1, frequency=VEHICLE_IO_RATE)
    if not car.card.is_valid():
        raise RuntimeError("La tarjeta QCar no abrio. Cierra otros procesos QCar/Simulink.")
    return car


def read_sensors(car, throttle, steering):
    car.read_write_std(throttle, steering)
    v = float(np.ravel(car.motorTach)[0])
    wz = float(np.ravel(car.gyroscope)[2])
    return v, wz


def probe():
    car = open_vehicle()
    try:
        print("Gira el carro sobre su eje vertical. wz debe cambiar (quieto ~ 0).")
        for _ in range(100):
            car.read_write_std(0, 0)
            try:
                v = float(np.ravel(car.motorTach)[0])
                gyro = np.ravel(car.gyroscope)
                print("v=%6.3f m/s | wz=%7.3f rad/s | gyro=%s" % (v, gyro[2], np.round(gyro, 3)))
            except AttributeError:
                print("Falta motorTach/gyroscope. Atributos disponibles:")
                print([a for a in dir(car) if not a.startswith("_")])
                break
            time.sleep(0.1)
    finally:
        car.read_write_std(0, 0)
        car.terminate()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def control_loop(shared, ctl_log):
    car = None
    try:
        car = open_vehicle()
        dt_nom = 1.0 / CONTROL_RATE
        print("Calibrando giroscopio (carro quieto)...")
        samples = []
        for _ in range(100):
            samples.append(read_sensors(car, 0, 0)[1])
            time.sleep(dt_nom)
        bias = float(np.mean(samples))
        while not shared.ready and not shared.stop.is_set():
            time.sleep(0.1)
        th = th_ref = dist = thr = 0.0
        turning, turns_seen = False, 0
        t0 = tprev = time.perf_counter()
        while not shared.stop.is_set():
            now = time.perf_counter()
            dt, tprev = now - tprev, now
            if MAX_RUN_S and now - t0 > MAX_RUN_S:
                print("Tope de tiempo alcanzado")
                break
            # Una jirafa mueve el rumbo de referencia 90 grados; el mismo control de
            # rumbo que mantiene la recta hace el giro y luego sigue recto en el nuevo rumbo.
            requests = shared.get_turn_requests()
            if requests > turns_seen:
                turns_seen = requests
                if not turning:
                    turning = True
                    th_ref = wrap(th_ref + TURN_DIR * TURN_ANGLE)
                    print("Jirafa: iniciando giro de %.0f grados" % math.degrees(TURN_DIR * TURN_ANGLE))
            err = wrap(th_ref - th)
            if turning and abs(err) < TURN_TOL:
                turning = False
                print("Giro completado: sigue en linea recta")
            steer = float(np.clip(STEER_SIGN * KP_HEADING * err, -STEER_MAX, STEER_MAX))
            factor, age, active = shared.get_events()
            if age > STALE_S:
                factor, state = 0.0, "SAFE_STOP"
            else:
                state = "+".join(active) if active else "CRUISE"
            if turning:
                state = "GIRO " + state
            target = THROTTLE_CRUISE * factor
            if target > 0:
                target = max(target, THROTTLE_MIN)
            step = (BRAKE_RATE if target < thr else ACCEL_RATE) * dt
            thr += float(np.clip(target - thr, -step, step))
            v, wz = read_sensors(car, thr, steer)
            th = wrap(th + (wz - bias) * dt)
            dist += abs(v) * dt
            shared.telemetry = dict(th=th, ref=th_ref, v=v, thr=thr, steer=steer, dist=dist,
                                    state=state)
            ctl_log.append((round(now - t0, 3), th, th_ref, v, dist, steer, thr, state, factor))
            time.sleep(max(0.0, dt_nom - (time.perf_counter() - now)))
    except Exception:
        traceback.print_exc()
    finally:
        if car is not None:
            car.read_write_std(0, 0)
            car.terminate()
            print("Carro detenido.")
        shared.stop.set()


# ------------------------------------------------------------------ MAIN
def dump_csv(name, header, rows):
    if not rows:
        return
    path = os.path.join(LOG_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("Guardado:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="imprime sensores, sin mover")
    ap.add_argument("--drive", action="store_true", help="habilita movimiento del carro")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelo no encontrado: %s" % MODEL_PATH)
    shared, events = Shared(), make_events()
    event_log, ctl_log = [], []
    ctl = None
    if args.drive:
        print("MOVIMIENTO HABILITADO, sin limite de distancia. Detener con Ctrl+C (o q).")
        ctl = threading.Thread(target=control_loop, args=(shared, ctl_log), daemon=True)
        ctl.start()
    else:
        print("Modo solo percepcion: el carro NO se mueve.")
    try:
        run_perception(shared, events, not args.no_display, event_log)
    except KeyboardInterrupt:
        print("\nCtrl+C: deteniendo todo...")
    finally:
        shared.stop.set()
        if ctl is not None:
            ctl.join(timeout=5)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dump_csv("events_%s.csv" % stamp, ["t_s", "evento", "estado"], event_log)
        dump_csv("run_%s.csv" % stamp,
                 ["t_s", "rumbo", "rumbo_ref", "v", "dist", "steer", "throttle", "estado",
                  "factor"],
                 ctl_log)


if __name__ == "__main__":
    main()

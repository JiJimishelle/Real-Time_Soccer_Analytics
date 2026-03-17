import argparse, math, time, os
from collections import Counter, deque, defaultdict
from statistics import mode

import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from sklearn.cluster import KMeans

COCO_PERSON = 0
BALL_CLASS  = 0

def cxcy(box):
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def crop_jersey_bbox(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None
    top = y1 + int(0.25 * h)
    bot = y1 + int(0.65 * h)
    left = x1 + int(0.20 * w)
    right = x2 - int(0.20 * w)
    if bot <= top or right <= left:
        return None
    return (left, top, right, bot)

def mask_non_jersey(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    skin1 = cv2.inRange(hsv, (0, 30, 60), (20, 200, 255))
    skin2 = cv2.inRange(hsv, (160, 30, 60), (179, 200, 255))
    skin = cv2.bitwise_or(skin1, skin2)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    keep_sat = cv2.inRange(S, 60, 255)
    keep_white = cv2.inRange(V, 200, 255)
    non_grass = cv2.bitwise_not(grass)
    non_skin = cv2.bitwise_not(skin)
    keep = cv2.bitwise_or(keep_sat, keep_white)
    mask = cv2.bitwise_and(non_grass, non_skin)
    mask = cv2.bitwise_and(mask, keep)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def is_referee_yellow(bgr_roi, mask=None):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    if mask is not None:
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    yellow = cv2.inRange(hsv, (20, 80, 80), (50, 255, 255))
    return cv2.countNonZero(yellow) > 150

def jersey_feature(bgr_roi, mask):
    if bgr_roi is None or mask is None or bgr_roi.size == 0:
        return None
    if cv2.countNonZero(mask) < 200:
        return None
    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
    Lch, Ach, Bch = cv2.split(lab)
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    Sch = hsv[:, :, 1]
    sel = mask > 0
    Lm = float(Lch[sel].mean()); Am = float(Ach[sel].mean())
    Bm = float(Bch[sel].mean()); Sm = float(Sch[sel].mean())
    return np.array([Lm, Am, Bm, Sm], dtype=np.float32)

class TeamColorClusterer:
    def __init__(self, warmup_min=30, ema=0.92, std_alpha=2.5):
        self.warmup_min = warmup_min
        self.ema = ema
        self.std_alpha = std_alpha
        self.init_feats = []
        self.centroids = None
        self.thresh = None
        self.ready = False

    def update_init(self, feat):
        if feat is not None:
            self.init_feats.append(feat)

    def initialize(self):
        if len(self.init_feats) < self.warmup_min:
            return False
        X = np.vstack(self.init_feats)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        C = km.cluster_centers_.astype(np.float32)
        labels = km.labels_
        dists = np.linalg.norm(X - C[labels], axis=1)
        thr = []
        for k in [0, 1]:
            dk = dists[labels == k]
            mu = float(dk.mean()) if len(dk) else 0.0
            sd = float(dk.std()) if len(dk) else 0.0
            thr.append(mu + self.std_alpha * sd if sd > 0 else mu + 5.0)
        self.centroids = C
        self.thresh = np.array(thr, dtype=np.float32)
        self.ready = True
        self.init_feats = []
        return True

    def predict(self, feat):
        if not self.ready or self.centroids is None or self.thresh is None:
            return "Unknown"
        d = np.linalg.norm(self.centroids - feat[None, :], axis=1)
        k = int(np.argmin(d))
        if d[k] > self.thresh[k]:
            return "Unknown"
        return "Team A" if k == 0 else "Team B"

    def ema_update(self, feat, label):
        if not self.ready:
            return
        k = 0 if label == "Team A" else (1 if label == "Team B" else None)
        if k is None:
            return
        self.centroids[k] = self.ema * self.centroids[k] + (1.0 - self.ema) * feat

def to_detections(person_boxes, person_confs):
    dets = []
    for (x1, y1, x2, y2), conf in zip(person_boxes, person_confs):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dets.append(Detection(points=np.array([[cx, cy]]), scores=np.array([float(conf)])))
    return dets

class ClusterInertiaLabeler:
    def __init__(self, distance_threshold=40, inertia_window=20, warmup_min=30, ema=0.92, std_alpha=2.5):
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=distance_threshold)
        self.history = defaultdict(lambda: deque(maxlen=inertia_window))
        self.clusterer = TeamColorClusterer(warmup_min=warmup_min, ema=ema, std_alpha=std_alpha)

    def labels_for(self, frame, person_boxes, person_confs):
        if not person_boxes:
            return [], []

        dets = to_detections(person_boxes, person_confs)
        tracked = self.tracker.update(dets)

        labels_for_idx = {}
        ids_for_idx = {}
        centers = [((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0) for bx in person_boxes]

        for t in tracked:
            tid = t.id
            cx, cy = t.estimate[0]
            j = int(np.argmin([abs(c[0] - cx) + abs(c[1] - cy) for c in centers])) if centers else None
            if j is None:
                continue

            crop = crop_jersey_bbox(person_boxes[j])
            if crop is None:
                cur = "Unknown"
            else:
                x1, y1, x2, y2 = crop
                roi = frame[y1:y2, x1:x2]
                mask = mask_non_jersey(roi)

                if is_referee_yellow(roi, mask):
                    cur = "Referee"
                else:
                    feat = jersey_feature(roi, mask)
                    if not self.clusterer.ready:
                        self.clusterer.update_init(feat)
                        cur = "Unknown"
                    else:
                        cur = self.clusterer.predict(feat) if feat is not None else "Unknown"
                        if cur in ("Team A", "Team B") and feat is not None:
                            self.clusterer.ema_update(feat, cur)

            self.history[tid].append(cur)
            votes = [l for l in self.history[tid] if l != "Unknown"]
            if votes:
                try:
                    final = mode(votes)
                except Exception:
                    final = votes[-1]
            else:
                final = cur

            labels_for_idx[j] = final
            ids_for_idx[j] = tid

        if not self.clusterer.ready:
            self.clusterer.initialize()

        labels_out, ids_out = [], []
        for i in range(len(person_boxes)):
            labels_out.append(labels_for_idx.get(i, "Unknown"))
            ids_out.append(ids_for_idx.get(i, -1))
        return labels_out, ids_out

def draw_possession_bar(frame, a_count, b_count, a_name="Team A", b_name="Team B"):
    h, w = frame.shape[:2]
    tot = max(1, a_count + b_count)
    aW = int(w * (a_count / tot))
    cv2.rectangle(frame, (0, h - 34), (aW, h - 6), (255, 0, 0), -1)
    cv2.rectangle(frame, (aW, h - 34), (w, h - 6), (255, 255, 255), -1)
    cv2.rectangle(frame, (0, h - 34), (w, h - 6), (40, 40, 40), 2)
    cv2.putText(frame, f"{a_name}: {a_count/tot*100:.1f}%", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"{b_name}: {b_count/tot*100:.1f}%", (w - 220, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ball_model", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--hold_ms", type=int, default=800)
    ap.add_argument("--save", default="possession_out.mp4")
    ap.add_argument("--person_conf", type=float, default=0.45)
    ap.add_argument("--ball_conf", type=float, default=0.20)
    ap.add_argument("--inertia", type=int, default=20)
    ap.add_argument("--warmup_min", type=int, default=40)
    ap.add_argument("--distance_threshold", type=float, default=40)
    ap.add_argument("--std_alpha", type=float, default=2.5)
    args = ap.parse_args()

    player_model = YOLO("yolov8n.pt")
    ball_model   = YOLO(args.ball_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"❌ Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    hold_frames = int((args.hold_ms / 1000.0) * fps)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    labeler = ClusterInertiaLabeler(distance_threshold=args.distance_threshold,
                                    inertia_window=args.inertia,
                                    warmup_min=args.warmup_min,
                                    ema=0.92,
                                    std_alpha=args.std_alpha)

    counts = Counter()
    last_team, hold = None, 0
    frame_i, t0 = 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pres = player_model.predict(frame, imgsz=args.imgsz, verbose=False)[0]
        bres = ball_model.predict(frame,   imgsz=args.imgsz, verbose=False)[0]

        persons, person_confs = [], []
        for xyxy, c, conf in zip(pres.boxes.xyxy, pres.boxes.cls, pres.boxes.conf):
            if int(c) == COCO_PERSON and conf > args.person_conf:
                persons.append(xyxy.cpu().numpy())
                person_confs.append(float(conf))

        balls = []
        for xyxy, c, conf in zip(bres.boxes.xyxy, bres.boxes.cls, bres.boxes.conf):
            if int(c) == BALL_CLASS and conf > args.ball_conf:
                balls.append(xyxy.cpu().numpy())

        # Unpack labels + IDs
        player_teams, player_ids = labeler.labels_for(frame, persons, person_confs)
        poss_teams = ["Unknown" if t == "Referee" else t for t in player_teams]

        ball_center = None
        if balls:
            best = max(balls, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            ball_center = cxcy(best)

        frame_team = None
        if ball_center and persons:
            bx, by = ball_center
            centers = [cxcy(pb) for pb in persons]
            dists = [math.hypot(cx - bx, cy - by) for (cx, cy) in centers]
            idx = int(np.argmin(dists))
            frame_team = poss_teams[idx]
            if frame_team not in ("Team A", "Team B"):
                frame_team = None

        if frame_team:
            last_team, hold = frame_team, 0
        elif last_team and hold < hold_frames:
            frame_team, hold = last_team, hold + 1

        if frame_team in ("Team A", "Team B"):
            counts[frame_team] += 1

        # Draw players with IDs
        for (pb, t, pid) in zip(persons, player_teams, player_ids):
            x1, y1, x2, y2 = map(int, pb)
            if t == "Team A":         color = (255, 0, 0)
            elif t == "Team B":       color = (255, 255, 255)
            elif t == "Referee":      color = (0, 255, 255)
            else:                     color = (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{t} | ID:{pid if pid>=0 else '-'}"
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if ball_center:
            bx, by = map(int, ball_center)
            cv2.circle(frame, (bx, by), 6, (0, 255, 0), -1)

        draw_possession_bar(frame, counts["Team A"], counts["Team B"], "Team A", "Team B")
        out.write(frame)
        frame_i += 1

        if frame_i % 120 == 0:
            fps_now = frame_i / (time.time() - t0)
            print(f"[{frame_i}] ~{fps_now:.1f} fps | A {counts['Team A']}  B {counts['Team B']} | "
                  f"cluster_ready={labeler.clusterer.ready}")
    # Save features for visualization
        if hasattr(labeler.clusterer, "init_feats") and len(labeler.clusterer.init_feats) > 0: 
            np.savez("kmeans_debug_feats.npz",
                X=np.vstack(labeler.clusterer.init_feats),
                y=np.array(["warmup"] * len(labeler.clusterer.init_feats)))
            print("✅ Saved debug features to kmeans_debug_feats.npz")

    cap.release(); out.release()

    total = max(1, counts["Team A"] + counts["Team B"])
    print("\n===== Possession Summary =====")
    print(f"Team A: {counts['Team A']/total*100:.1f}%")
    print(f"Team B: {counts['Team B']/total*100:.1f}%")
    print(f"💾 Saved video -> {os.path.abspath(args.save)}")
    print("Done.")

if __name__ == "__main__":
    main()

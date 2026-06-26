"""
ADAS Core Engine
================
Headless, fast object detection + tracking + collision/lane alerting.
No Streamlit. No dashboard. This is the engine — wrap it in a UI later.

Usage:
    python adas_engine.py --source videos/test.mp4 --weights models/best.pt
    python adas_engine.py --source 0 --weights models/best.pt          # webcam
    python adas_engine.py --source videos/test.mp4 --weights models/best.pt --headless --save outputs/result.mp4

Design choices (why this is fast):
- Loads YOUR trained weights, classes pulled directly from the model (no hardcoded COCO indices).
- Single model.track() call per frame — no duplicate inference.
- Lane detection (Hough) runs every LANE_DETECT_EVERY_N frames, not every frame; reused/smoothed in between.
- All drawing uses cv2 primitives (fast, C-backed), not per-pixel PIL loops.
- No BEV/chase-view rendering in the engine (it was the heaviest non-model cost) — moved to optional UI layer.
- Real FPS counter shown on screen so you can measure, not guess.
"""

import argparse
import time
import json
import os
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Shared control flags — can be flipped from server.py at runtime
# ---------------------------------------------------------------------------
_lanes_enabled = threading.Event()
_lanes_enabled.set()   # lanes ON by default

def toggle_lanes():
    """Flip lane detection on/off. Returns the new state (True = enabled)."""
    if _lanes_enabled.is_set():
        _lanes_enabled.clear()
        return False
    else:
        _lanes_enabled.set()
        return True

def lanes_are_enabled() -> bool:
    return _lanes_enabled.is_set()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

FRAME_SIZE = (854, 480)          # processing resolution (resize down from source for speed)
CONF_THRESHOLD = 0.20           # lower than 0.45 — your model is still undertrained (25 epochs);
                                  # raise this back up once retrained to 150 epochs and re-validated
COLLISION_DISTANCE_M = 15.0
LANE_DEVIATION_PX = 30
LANE_DETECT_EVERY_N = 4          # run Hough lane detection every Nth frame only
TRACK_HISTORY_LEN = 10

# Real-world widths (m) for monocular distance estimation — used by class name (case-insensitive match)
VEHICLE_WIDTHS_M = {
    "car": 1.8, "truck": 2.5, "bus": 3.0, "two wheeler": 0.8,
    "three wheeler": 1.4, "tractor": 2.0, "person": 0.5, "fixed obstacle": 1.0,
    "pothole": 0.5, "auto": 1.4, "rikshaw": 1.4, "carts": 1.2, "animal": 0.6
}
ASSUMED_FOCAL_LENGTH = 750  # px — recalibrate against your actual camera if doing real distance work

COLORS = {
    "critical": (0, 30, 220),
    "in_lane": (0, 165, 255),
    "off_lane": (180, 180, 40),
    "lane_line": (50, 230, 50),
    "hud_bg": (18, 18, 18),
}


# ----------------------------------------------------------------------------
# Lane detection (throttled)
# ----------------------------------------------------------------------------

def trapezoid_roi_mask(image, horizon_ratio=0.55, hood_ratio=1.0, top_spread=0.10, bottom_spread=0.90):
    h, w = image.shape[:2]
    mask = np.zeros_like(image)
    top_y, bottom_y = int(h * horizon_ratio), int(h * hood_ratio)
    cx = w // 2
    top_half, bottom_half = int((w * top_spread) / 2), int((w * bottom_spread) / 2)
    trapezoid = np.array([[
        (cx - bottom_half, bottom_y), (cx - top_half, top_y),
        (cx + top_half, top_y), (cx + bottom_half, bottom_y),
    ]])
    cv2.fillPoly(mask, trapezoid, 255)
    return cv2.bitwise_and(image, image, mask=mask)


def find_lane_segments(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    roi = trapezoid_roi_mask(edges)
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=30,
                             minLineLength=40, maxLineGap=150)

    left_candidates, right_candidates = [], []
    width = frame.shape[1]
    left_limit, right_limit = width * 0.85, width * 0.15
    if lines is None:
        return left_candidates, right_candidates

    for (x1, y1, x2, y2), in lines.reshape(-1, 1, 4):
        if x1 == x2:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        length = np.hypot(x2 - x1, y2 - y1)
        if -2.5 < slope < -0.5 and x1 < left_limit and x2 < left_limit:
            left_candidates.append((slope, intercept, length))
        elif 0.5 < slope < 2.5 and x1 > right_limit and x2 > right_limit:
            right_candidates.append((slope, intercept, length))
    return left_candidates, right_candidates


def weighted_average_line(candidates):
    weights = [c[2] for c in candidates]
    return np.average(candidates, axis=0, weights=weights)


class LaneState:
    """Tracks smoothed left/right lane lines across frames. Updated every Nth frame."""

    def __init__(self, history_len=TRACK_HISTORY_LEN):
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)
        self.frame_count = 0

    def maybe_update(self, frame):
        self.frame_count += 1
        if self.frame_count % LANE_DETECT_EVERY_N != 0:
            return
        left_candidates, right_candidates = find_lane_segments(frame)
        if left_candidates:
            self.left_history.append(weighted_average_line(left_candidates))
        if right_candidates:
            self.right_history.append(weighted_average_line(right_candidates))

    def current_lines(self):
        if not self.left_history or not self.right_history:
            return None
        left = np.mean(self.left_history, axis=0)
        right = np.mean(self.right_history, axis=0)
        return left[0], left[1], right[0], right[1]  # m1, b1, m2, b2


def build_lane_polygon(lane_eqs, height):
    m1, b1, m2, b2 = lane_eqs
    y1 = height
    lx1, rx1 = int((y1 - b1) / m1), int((y1 - b2) / m2)
    
    # Fix: Prevent lane from becoming a thin narrow polygon
    min_width = 250  # roughly 30% of screen width
    if rx1 - lx1 < min_width:
        center = (lx1 + rx1) // 2
        lx1 = center - min_width // 2
        rx1 = center + min_width // 2

    target_y2 = int(height * 0.60)
    if abs(m1 - m2) > 1e-3:
        ix = (b2 - b1) / (m1 - m2)
        iy = m1 * ix + b1
        if iy > target_y2:
            target_y2 = int(iy + 25)
    target_y2 = min(target_y2, int(height * 0.80))
    ly2 = ry2 = target_y2
    lx2, rx2 = int((ly2 - b1) / m1), int((ry2 - b2) / m2)
    polygon = np.array([[lx1, y1], [lx2, ly2], [rx2, ry2], [rx1, y1]], dtype=np.int32)
    points = {"lx1": lx1, "ly1": y1, "lx2": lx2, "ly2": ly2, "rx1": rx1, "ry1": y1, "rx2": rx2, "ry2": ry2}
    return polygon, points


def classify_road_state(lane_eqs, polygon_points, img_center):
    m1, _, m2, _ = lane_eqs
    turn_status = "Straight Road"
    if abs(m1) < 0.7 and m2 < 1.0:
        turn_status = "CURVE AHEAD: LEFT <"
    elif abs(m1) > 1.3 and m2 > 0.7:
        turn_status = "CURVE AHEAD: RIGHT >"

    lane_center = (polygon_points["lx1"] + polygon_points["rx1"]) // 2
    deviation = lane_center - img_center
    if deviation < -LANE_DEVIATION_PX:
        departure_status = "! DEPARTING RIGHT !"
    elif deviation > LANE_DEVIATION_PX:
        departure_status = "! DEPARTING LEFT !"
    else:
        departure_status = "LANE KEEP ASSIST: OK"
    return turn_status, departure_status


def draw_lane_overlay(frame, polygon, polygon_points):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (0, 160, 240))
    frame = cv2.addWeighted(frame, 0.95, overlay, 0.05, 0)
    p = polygon_points
    cv2.line(frame, (p["lx1"], p["ly1"]), (p["lx2"], p["ly2"]), COLORS["lane_line"], 5)
    cv2.line(frame, (p["rx1"], p["ry1"]), (p["rx2"], p["ry2"]), COLORS["lane_line"], 5)
    cv2.line(frame, (p["lx2"], p["ly2"]), (p["rx2"], p["ry2"]), COLORS["lane_line"], 3)
    return frame


# ----------------------------------------------------------------------------
# Distance + detection helpers
# ----------------------------------------------------------------------------

def estimate_distance_m(pixel_width, class_name):
    clean_name = class_name.lower().replace("-", " ")
    real_width = VEHICLE_WIDTHS_M.get(clean_name, 1.5)
    pixel_width = max(pixel_width, 1)
    return (real_width * ASSUMED_FOCAL_LENGTH) / pixel_width


def is_inside_lane(box_x, y1, y2, box_height, lane_polygon):
    p1, p2 = (box_x, y2), (box_x, int(y1 + box_height * 0.75))
    return any(cv2.pointPolygonTest(lane_polygon, p, False) >= 0 for p in (p1, p2))


# ----------------------------------------------------------------------------
# Fast cv2-only drawing (replaces the old PIL per-pixel pill drawing)
# ----------------------------------------------------------------------------

def draw_premium_box(img, x1, y1, x2, y2, color, thickness=1, corner_thickness=3):
    # Dynamic corner length based on box size to avoid overlap
    corner_len = min(12, (x2 - x1) // 3, (y2 - y1) // 3)
    corner_len = max(3, corner_len)
    
    # Draw main thin box border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, corner_thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, corner_thickness, cv2.LINE_AA)
    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, corner_thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, corner_thickness, cv2.LINE_AA)
    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, corner_thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, corner_thickness, cv2.LINE_AA)
    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, corner_thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, corner_thickness, cv2.LINE_AA)


def draw_label(frame, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.38, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 6, 4
    
    # Place label above the box if space permits, else place it inside
    y_offset = -5
    if y - th - pad_y * 2 < 0:
        y_offset = th + pad_y * 2 + 5
        
    bx1 = x
    by1 = y + y_offset - th - pad_y
    bx2 = min(frame.shape[1] - 1, x + tw + pad_x * 2)
    by2 = y + y_offset + pad_y
    
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + pad_x, by2 - pad_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def annotate_detections(frame, results, model_names, lane_polygon):
    detections = []
    collision_alert = False
    if results is None:
        return detections, collision_alert

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return detections, collision_alert

    has_ids = boxes.id is not None
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        class_name = model_names[int(boxes.cls[i])]
        
        # Heuristic fix: if it's labeled "person" but proportions are wrong
        if class_name.lower() == "person":
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width > box_height * 1.5:
                # Very wide/flat -> pothole
                class_name = "pothole"
            elif box_width > box_height * 0.8:
                # Square-ish or moderately wide -> animal (like a dog)
                class_name = "Animal"
                
        conf = float(boxes.conf[i])
        track_id = int(boxes.id[i]) if has_ids else -1

        distance_m = estimate_distance_m(x2 - x1, class_name)
        label = f"{class_name.upper()} #{track_id} [{distance_m:.1f}m]" if track_id >= 0 else f"{class_name.upper()} [{conf:.2f}]"

        center_x, foot_y, box_height = (x1 + x2) // 2, y2, y2 - y1
        is_critical = False
        box_color = COLORS["off_lane"]

        is_in_lane = False
        if lane_polygon is not None and is_inside_lane(center_x, y1, y2, box_height, lane_polygon):
            is_in_lane = True
        elif lane_polygon is None:
            # Fallback: if lane detection is disabled, assume a central driving corridor (30% of frame width)
            img_center = frame.shape[1] // 2
            if abs(center_x - img_center) < frame.shape[1] * 0.15:
                is_in_lane = True

        if is_in_lane:
            if distance_m < COLLISION_DISTANCE_M:
                collision_alert, is_critical = True, True
                box_color = COLORS["critical"]
                draw_premium_box(frame, x1, y1, x2, y2, box_color, thickness=1, corner_thickness=3)
                draw_label(frame, f"CRITICAL: {label}", x1, y1, box_color)
            else:
                box_color = COLORS["in_lane"]
                draw_premium_box(frame, x1, y1, x2, y2, box_color, thickness=1, corner_thickness=2)
                draw_label(frame, label, x1, y1, box_color)
        else:
            box_color = COLORS["off_lane"]
            draw_premium_box(frame, x1, y1, x2, y2, box_color, thickness=1, corner_thickness=2)
            draw_label(frame, label, x1, y1, box_color, text_color=(255, 255, 255))

        # Render translucent highlight overlay inside the bounding box
        h_img, w_img = frame.shape[:2]
        x1_c, y1_c = max(0, min(x1, w_img - 1)), max(0, min(y1, h_img - 1))
        x2_c, y2_c = max(0, min(x2, w_img - 1)), max(0, min(y2, h_img - 1))
        if x2_c > x1_c and y2_c > y1_c:
            overlay = frame[y1_c:y2_c, x1_c:x2_c].copy()
            cv2.rectangle(overlay, (0, 0), (x2_c - x1_c, y2_c - y1_c), box_color, -1)
            cv2.addWeighted(overlay, 0.15, frame[y1_c:y2_c, x1_c:x2_c], 0.85, 0, dst=frame[y1_c:y2_c, x1_c:x2_c])

        detections.append((class_name, center_x, foot_y, is_critical, distance_m, track_id, is_in_lane))

    return detections, collision_alert


def draw_hud(frame, turn_status, departure_status, collision_alert, fps, object_count=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Sleek floating HUD top card - slightly taller and darker
    hud_y1, hud_y2 = 10, 65
    
    # Isolate blending to ROI only to prevent any full-frame exposure shifts
    roi = frame[hud_y1:hud_y2, 15:w-15]
    overlay_roi = roi.copy()
    cv2.rectangle(overlay_roi, (0, 0), (w - 30, hud_y2 - hud_y1), (10, 10, 15), -1)
    cv2.addWeighted(overlay_roi, 0.8, roi, 0.2, 0, dst=roi)
    frame[hud_y1:hud_y2, 15:w-15] = roi
    
    cv2.rectangle(frame, (15, hud_y1), (w - 15, hud_y2), (100, 100, 100), 1, cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Divide the HUD width into 4 equal columns
    usable_w = (w - 30)
    col_w = usable_w // 4
    centers = [15 + col_w // 2 + i * col_w for i in range(4)]
    
    # Draw vertical dividers
    for i in range(1, 4):
        x_div = 15 + i * col_w
        cv2.line(frame, (x_div, hud_y1 + 10), (x_div, hud_y2 - 10), (100, 100, 100), 1, cv2.LINE_AA)

    def draw_centered_text(img, text, cx, y, font_scale, color, thickness):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(img, text, (cx - tw // 2, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Column 1: ADAS STATUS
    draw_centered_text(frame, "ADAS SYSTEM", centers[0], hud_y1 + 18, 0.4, (180, 180, 180), 1)
    if collision_alert:
        draw_centered_text(frame, "COLLISION ALERT", centers[0], hud_y1 + 40, 0.5, (0, 30, 255), 2)
    else:
        draw_centered_text(frame, "ACTIVE / OK", centers[0], hud_y1 + 40, 0.5, (50, 240, 50), 1)
        
    # Column 2: LANE ASSIST
    draw_centered_text(frame, "LANE ASSIST", centers[1], hud_y1 + 18, 0.4, (180, 180, 180), 1)
    dep_color = (0, 30, 255) if "!" in departure_status else (50, 240, 50)
    draw_centered_text(frame, departure_status, centers[1], hud_y1 + 40, 0.5, dep_color, 1)
    
    # Column 3: ROAD GEOMETRY
    draw_centered_text(frame, "ROAD GEOMETRY", centers[2], hud_y1 + 18, 0.4, (180, 180, 180), 1)
    draw_centered_text(frame, turn_status, centers[2], hud_y1 + 40, 0.5, (255, 180, 0), 1)

    # Column 4: SYSTEM METRICS
    draw_centered_text(frame, "SYSTEM METRICS", centers[3], hud_y1 + 18, 0.4, (180, 180, 180), 1)
    metrics_text = f"FPS: {fps:.1f} | OBJ: {object_count}"
    draw_centered_text(frame, metrics_text, centers[3], hud_y1 + 40, 0.45, (180, 180, 40), 1)

    # Critical Flashing Overlay
    if collision_alert:
        # Full frame alert border
        cv2.rectangle(frame, (0, 0), (w, h), (0, 30, 255), 6)
        
        # Center collision warning card
        card_w, card_h = 300, 40
        cx1, cy1 = (w - card_w) // 2, int(h * 0.78)
        cx2, cy2 = cx1 + card_w, cy1 + card_h
        
        warning_overlay = frame.copy()
        cv2.rectangle(warning_overlay, (cx1, cy1), (cx2, cy2), (0, 0, 200), -1)
        frame = cv2.addWeighted(frame, 0.6, warning_overlay, 0.4, 0)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 30, 255), 2, cv2.LINE_AA)
        
        w_text = "CRITICAL: BRAKE NOW"
        draw_centered_text(frame, w_text, cx1 + card_w // 2, cy1 + 26, 0.6, (255, 255, 255), 2)
        
    return frame



def draw_bev_visualizer(frame, detections, lane_detected, departure_status, turn_status, deviation=None):
    h, w = frame.shape[:2]
    widget_w, widget_h = 160, 180
    pad_x, pad_y = 20, 20
    bx1, by1 = w - widget_w - pad_x, h - widget_h - pad_y
    bx2, by2 = w - pad_x, h - pad_y
    widget_cx = bx1 + widget_w // 2
    
    # 0. Define mapping scales to prevent overlapping the ego vehicle
    bev_zero_y = by2 - 45       # Ground 0m is 45px above widget bottom, giving ego an 8px buffer
    bev_horizon_y = by1 + 25    # Far horizon mapping limit
    
    # 1. Overlay for translucent card and vehicle bodies
    roi = frame[by1:by2, bx1:bx2]
    overlay_roi = roi.copy()
    
    # Background glassmorphic card
    cv2.rectangle(overlay_roi, (0, 0), (widget_w, widget_h), (18, 18, 18), -1)
    
    # Ego vehicle coordinates (3D Cuboid)
    ego_y = by2 - 25
    ep1 = (int(widget_cx - 7), int(ego_y + 12))
    ep2 = (int(widget_cx + 7), int(ego_y + 12))
    ep3 = (int(widget_cx + 7), int(ego_y - 12))
    ep4 = (int(widget_cx - 7), int(ego_y - 12))
    ep5 = (ep1[0], ep1[1] - 12)
    ep6 = (ep2[0], ep2[1] - 12)
    ep7 = (ep3[0], ep3[1] - 12)
    ep8 = (ep4[0], ep4[1] - 12)
    
    ego_color = (240, 160, 0)  # Cyan-blue translucent body
    ego_faces = [
        np.array([ep1, ep2, ep3, ep4], dtype=np.int32),
        np.array([ep5, ep6, ep7, ep8], dtype=np.int32),
        np.array([ep1, ep2, ep6, ep5], dtype=np.int32),
        np.array([ep3, ep4, ep8, ep7], dtype=np.int32),
        np.array([ep1, ep4, ep8, ep5], dtype=np.int32),
        np.array([ep2, ep3, ep7, ep6], dtype=np.int32)
    ]
    for face in ego_faces:
        # Offset for ROI
        roi_face = np.array([[p[0] - bx1, p[1] - by1] for p in face], dtype=np.int32)
        cv2.fillPoly(overlay_roi, [roi_face], ego_color)
        
    # Real-world dimensions (width, length, height) in meters for Tesla BEV 3D mapping
    VEHICLE_DIMS = {
        "car": (1.8, 4.0, 1.4),
        "truck": (2.5, 7.5, 2.8),
        "bus": (3.0, 10.0, 3.0),
        "two wheeler": (0.8, 1.8, 1.3),
        "three wheeler": (1.4, 2.8, 1.6),
        "tractor": (2.0, 4.5, 2.0),
        "person": (0.5, 0.5, 1.7),
        "fixed obstacle": (1.0, 1.0, 1.2),
        "pothole": (0.8, 0.8, 0.1),
        "auto": (1.4, 2.8, 1.6),
        "rikshaw": (1.4, 2.8, 1.6),
        "carts": (1.2, 2.5, 1.5),
        "animal": (0.6, 1.5, 1.0),
    }

    # Classes that shouldn't be drawn on the 3D BEV radar (lights, signs, etc)
    NON_VEHICLE_CLASSES = {
        "green_light", "red_light", "yellow_light", "traffic_light", 
        "warning", "stop", "do_not_enter", "do_not_stop", "do_not_turn_l",
        "do_not_turn_r", "do_not_u_turn", "enter_left_lane", "left_right_lane",
        "no_parking", "ped_zebra_cross", "railway_crossing", "t_intersection_l", "2"
    }

    # 2. Render surrounding traffic vehicle bodies on overlay
    img_center = w // 2
    for class_name, center_x, foot_y, is_critical, distance_m, track_id, is_in_lane in detections:
        # Distance checks
        if distance_m <= 0 or distance_m > 45.0 or class_name.lower() in NON_VEHICLE_CLASSES:
            continue
        # Estimate lateral distance (m) using pinhole model
        x_m = (center_x - img_center) * distance_m / ASSUMED_FOCAL_LENGTH
        
        # Map distance to relative Y in widget (0 to 45m -> bev_horizon_y to bev_zero_y)
        y_rel = (distance_m / 45.0) * (bev_zero_y - bev_horizon_y)
        vy = int(bev_zero_y - y_rel)
        
        # Map lateral offset to relative X in widget (-6m to +6m -> -70px to +70px)
        x_rel = (x_m / 6.0) * 70
        vx = int(widget_cx + x_rel)
        
        # Ensure center is safely within widget card boundary
        if by1 + 5 < vy < by2 - 5 and bx1 + 5 < vx < bx2 - 5:
            w_m, l_m, h_m = VEHICLE_DIMS.get(class_name.lower().replace("-", " "), (1.5, 3.5, 1.4))
            
            # Perspective scale
            scale = 1.0 - (distance_m / 52.0)
            scale = max(0.15, min(scale, 1.0))
            
            w_px = max(3, int(w_m * 7.5 * scale))
            l_px = max(4, int(l_m * 3.5 * scale))
            h_px = max(2, int(h_m * 6.5 * scale))
            
            # Vertices - Position the closest part (rear of box) at vy
            p1 = (int(vx - w_px // 2), int(vy))
            p2 = (int(vx + w_px // 2), int(vy))
            p3 = (int(vx + w_px // 2), int(vy - l_px))
            p4 = (int(vx - w_px // 2), int(vy - l_px))
            p5 = (p1[0], p1[1] - h_px)
            p6 = (p2[0], p2[1] - h_px)
            p7 = (p3[0], p3[1] - h_px)
            p8 = (p4[0], p4[1] - h_px)
            
            # Select color based on status
            if is_critical:
                v_color = COLORS["critical"]
            elif is_in_lane:
                v_color = COLORS["in_lane"]
            else:
                v_color = COLORS["off_lane"]
                
            faces = [
                np.array([p1, p2, p3, p4], dtype=np.int32),
                np.array([p5, p6, p7, p8], dtype=np.int32),
                np.array([p1, p2, p6, p5], dtype=np.int32),
                np.array([p3, p4, p8, p7], dtype=np.int32),
                np.array([p1, p4, p8, p5], dtype=np.int32),
                np.array([p2, p3, p7, p6], dtype=np.int32)
            ]
            for face in faces:
                # Need to offset face coordinates for ROI
                roi_face = np.array([[p[0] - bx1, p[1] - by1] for p in face], dtype=np.int32)
                cv2.fillPoly(overlay_roi, [roi_face], v_color)
            
    # Apply alpha blending for translucent shapes within ROI only
    cv2.addWeighted(overlay_roi, 0.35, roi, 0.65, 0, dst=roi)
    frame[by1:by2, bx1:bx2] = roi
    
    # 3. Draw solid sharp outlines, lines, and texts directly on the frame
    # Card outer border
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), 1, cv2.LINE_AA)
    
    # Ego vehicle 3D outlines
    cv2.line(frame, ep1, ep2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep2, ep3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep3, ep4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep4, ep1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep5, ep6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep6, ep7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep7, ep8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep8, ep5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep1, ep5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep2, ep6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep3, ep7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, ep4, ep8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 4. Lane boundaries in perspective
    shift_px = 0
    if lane_detected and deviation is not None:
        shift_px = int(deviation * 0.25)
        shift_px = max(-25, min(shift_px, 25))
        
    curve_shift = 0
    if "LEFT" in turn_status:
        curve_shift = -15
    elif "RIGHT" in turn_status:
        curve_shift = 15
        
    ly1, ly2 = by2 - 10, by1 + 25
    lx1 = widget_cx - 28 - shift_px
    lx2 = widget_cx - 14 - shift_px + curve_shift
    rx1 = widget_cx + 28 - shift_px
    rx2 = widget_cx + 14 - shift_px + curve_shift
    
    left_color = COLORS["lane_line"]
    right_color = COLORS["lane_line"]
    if departure_status is not None:
        if "DEPARTING LEFT" in departure_status:
            left_color = COLORS["critical"]
        elif "DEPARTING RIGHT" in departure_status:
            right_color = COLORS["critical"]
            
    cv2.line(frame, (lx1, ly1), (lx2, ly2), left_color, 2, cv2.LINE_AA)
    cv2.line(frame, (rx1, ly1), (rx2, ly2), right_color, 2, cv2.LINE_AA)
    
    # 5. UI text and grid lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "BEV RADAR", (bx1 + 10, by1 + 18), font, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    
    # 20m grid line (mapped to vehicle distance scale)
    y20 = int(bev_zero_y - (20.0 / 45.0) * (bev_zero_y - bev_horizon_y))
    cv2.line(frame, (bx1 + 8, y20), (bx2 - 8, y20), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "20m", (bx1 + 8, y20 - 3), font, 0.28, (120, 120, 120), 1, cv2.LINE_AA)
    
    # 40m grid line (mapped to vehicle distance scale)
    y40 = int(bev_zero_y - (40.0 / 45.0) * (bev_zero_y - bev_horizon_y))
    cv2.line(frame, (bx1 + 8, y40), (bx2 - 8, y40), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "40m", (bx1 + 8, y40 - 3), font, 0.28, (120, 120, 120), 1, cv2.LINE_AA)
    
    # Also draw outlines for detected traffic on the frame
    for class_name, center_x, foot_y, is_critical, distance_m, track_id, is_in_lane in detections:
        if distance_m <= 0 or distance_m > 45.0 or class_name.lower() in NON_VEHICLE_CLASSES:
            continue
        x_m = (center_x - img_center) * distance_m / ASSUMED_FOCAL_LENGTH
        y_rel = (distance_m / 45.0) * (bev_zero_y - bev_horizon_y)
        vy = int(bev_zero_y - y_rel)
        x_rel = (x_m / 6.0) * 70
        vx = int(widget_cx + x_rel)
        
        if by1 + 5 < vy < by2 - 5 and bx1 + 5 < vx < bx2 - 5:
            w_m, l_m, h_m = VEHICLE_DIMS.get(class_name.lower().replace("-", " "), (1.5, 3.5, 1.4))
            scale = 1.0 - (distance_m / 52.0)
            scale = max(0.15, min(scale, 1.0))
            
            w_px = max(3, int(w_m * 7.5 * scale))
            l_px = max(4, int(l_m * 3.5 * scale))
            h_px = max(2, int(h_m * 6.5 * scale))
            
            p1 = (int(vx - w_px // 2), int(vy))
            p2 = (int(vx + w_px // 2), int(vy))
            p3 = (int(vx + w_px // 2), int(vy - l_px))
            p4 = (int(vx - w_px // 2), int(vy - l_px))
            p5 = (p1[0], p1[1] - h_px)
            p6 = (p2[0], p2[1] - h_px)
            p7 = (p3[0], p3[1] - h_px)
            p8 = (p4[0], p4[1] - h_px)
            
            if is_critical:
                v_color = COLORS["critical"]
            elif is_in_lane:
                v_color = COLORS["in_lane"]
            else:
                v_color = COLORS["off_lane"]
                
            # Draw outlines
            cv2.line(frame, p1, p2, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p3, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p4, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p4, p1, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p5, p6, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p6, p7, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p7, p8, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p8, p5, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p1, p5, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p6, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p7, v_color, 1, cv2.LINE_AA)
            cv2.line(frame, p4, p8, v_color, 1, cv2.LINE_AA)
            
    return frame


# ----------------------------------------------------------------------------
# Frame pipeline
# ----------------------------------------------------------------------------

def process_frame(frame, model, lane_state, class_filter, conf, iou, fps, run_inference=True, cached_results=None, hood_height=0.15, imgsz=640, augment=False, enable_lanes=True):
    frame = cv2.resize(frame, FRAME_SIZE)
    height, width = frame.shape[:2]
    img_center = width // 2

    results = None
    if run_inference or cached_results is None:
        # Create a copy for YOLO inference to apply hood masking
        inference_frame = frame.copy()
            
        if hood_height > 0:
            mask_y = int(height * (1.0 - hood_height))
            inference_frame[mask_y:, :] = 0
            
        results = model.track(inference_frame, classes=class_filter, conf=conf, iou=iou, imgsz=imgsz, persist=True,
                               tracker="bytetrack.yaml", verbose=False, augment=augment)[0]
    else:
        results = cached_results

    # Update lane detection (updated internally based on frame interval)
    if enable_lanes:
        lane_state.maybe_update(frame)
        lane_eqs = lane_state.current_lines()
    else:
        lane_eqs = None

    lane_polygon = None
    turn_status = "Straight Road"
    departure_status = "System Ready" if enable_lanes else "Lanes Disabled"

    deviation = None
    if lane_eqs:
        lane_polygon, polygon_points = build_lane_polygon(lane_eqs, height)
        frame = draw_lane_overlay(frame, lane_polygon, polygon_points)
        turn_status, departure_status = classify_road_state(lane_eqs, polygon_points, img_center)
        lane_center = (polygon_points["lx1"] + polygon_points["rx1"]) // 2
        deviation = lane_center - img_center

    detections, collision_alert = annotate_detections(frame, results, model.names, lane_polygon)
    frame = draw_hud(frame, turn_status, departure_status, collision_alert, fps, object_count=len(detections))
    frame = draw_bev_visualizer(frame, detections, lane_eqs is not None, departure_status, turn_status, deviation)

    stats = {
        "turn_status": turn_status,
        "departure_status": departure_status,
        "collision_alert": collision_alert,
        "object_count": len(detections),
        "nearest_m": min((d[4] for d in detections), default=None),
        "lane_detected": lane_eqs is not None,
    }
    return frame, detections, stats, results


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def get_args_parser():
    config = {}
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"[Engine] Successfully loaded config.json: {config}")
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    default_source = config.get("source", "0")
    default_weights = config.get("weights", "models/best.pt")
    default_iou = config.get("iou", 0.75) # Higher IoU threshold prevents merging heavily occluded vehicles

    parser = argparse.ArgumentParser(description="ADAS core detection engine")
    parser.add_argument("--source", default=default_source, help="Video file path, or 0 for webcam")
    parser.add_argument("--weights", default=default_weights, help="Path to trained .pt or .onnx weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=default_iou, help="NMS IoU threshold (higher = less merging of dense traffic)")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated class indices to detect (default: all)")
    parser.add_argument("--headless", action="store_true", help="Don't open a display window")
    parser.add_argument("--save", type=str, default="", help="Path to save annotated output video")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    parser.add_argument("--hood-height", type=float, default=0.0, help="Bottom portion of screen to mask out (0.0 to 1.0)")
    parser.add_argument("--inference-every", type=int, default=2, help="Run YOLO inference every N frames (1 = every frame)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--augment", action="store_true", help="Enable test time augmentation for better detection (slower)")
    parser.add_argument("--disable-lanes", action="store_true", help="Start with lane detection disabled")
    return parser

def run_engine(args, callback=None, stop_event=None):

    model = YOLO(args.weights)
    print(f"Loaded model: {args.weights}")
    print(f"Classes: {model.names}")

    class_filter = None
    if args.classes:
        class_filter = [int(c) for c in args.classes.split(",")]
    else:
        # Exclude 'text' (index 9) class by default
        class_filter = [i for i, name in model.names.items() if name.lower() != "text"]

    source = int(args.source) if str(args.source).isdigit() else args.source
    print(f"[Engine] Attempting to open video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source: {source}")
        return

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, FRAME_SIZE)

    lane_state = LaneState()
    fps_smooth = deque(maxlen=30)
    frame_idx = 0
    cached_results = None
    # Sync the shared event with the CLI flag on startup
    if args.disable_lanes:
        _lanes_enabled.clear()
    else:
        _lanes_enabled.set()
    enable_lanes = _lanes_enabled.is_set()
    
    recent_red_light = 0
    chime_cooldown = 0

    while cap.isOpened():
        if stop_event is not None and stop_event.is_set():
            print("[Engine] Stop event received. Shutting down loop.")
            break
            
        t0 = time.time()
        ok, raw_frame = cap.read()
        if not ok:
            break

        fps_est = (1.0 / np.mean(fps_smooth)) if fps_smooth else 0.0
        
        # Decide if we run YOLO inference on this frame
        run_inference = (frame_idx % args.inference_every == 0)
        
        frame, detections, stats, new_results = process_frame(
            raw_frame, model, lane_state, class_filter, args.conf, args.iou, fps_est,
            run_inference=run_inference,
            cached_results=cached_results,
            hood_height=args.hood_height,
            imgsz=args.imgsz,
            augment=args.augment,
            enable_lanes=enable_lanes
        )
        
        if run_inference:
            cached_results = new_results

        # Traffic light chime logic
        green_light_chime = False
        if new_results is not None and new_results.boxes is not None:
            classes = new_results.boxes.cls.cpu().numpy()
            if 26 in classes: # red_light
                recent_red_light = frame_idx
            elif 17 in classes: # green_light
                if frame_idx - recent_red_light < 60 and frame_idx > chime_cooldown:
                    green_light_chime = True
                    chime_cooldown = frame_idx + 100
                    recent_red_light = 0
                    
        stats['green_light_chime'] = green_light_chime

        dt = time.time() - t0
        fps_smooth.append(dt)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF if not args.headless else -1
        if key == ord("q"):
            break
        elif key == ord("l"):
            enable_lanes = toggle_lanes()
            print(f"Lane detection set to: {'ON' if enable_lanes else 'OFF'}")

        # Also respect any external toggle from the web server
        enable_lanes = _lanes_enabled.is_set()

        if callback:
            callback(frame_idx, stats, detections, frame, key)

        if writer is not None:
            writer.write(frame)
        if not args.headless:
            cv2.imshow("ADAS Engine", frame)

        if args.max_frames and frame_idx >= args.max_frames:
            break

        if frame_idx % 30 == 0:
            print(f"frame {frame_idx} | fps {fps_est:.1f} | objects {stats['object_count']} | "
                  f"nearest {stats['nearest_m']} | collision {stats['collision_alert']}")

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    avg_fps = (1.0 / np.mean(fps_smooth)) if fps_smooth else 0.0
    print(f"\nDone. Processed {frame_idx} frames. Avg FPS: {avg_fps:.2f}")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    run_engine(args)

if __name__ == "__main__":
    main()
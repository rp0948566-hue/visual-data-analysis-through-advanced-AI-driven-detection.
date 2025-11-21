import cv2 
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import json
import time

# ==================== CONFIGURATION ====================
OUTPUT_DIR = "output_interactive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_INPUT = r"input\Highway-2.mp4"
YOLO_MODEL = r"yolov10x.pt"
CONFIDENCE = 0.3

# Object colors (BGR format)
COLORS = {
    'car': (0, 255, 255),        # Cyan
    'truck': (255, 0, 255),      # Magenta
    'motorcycle': (0, 255, 0),   # Green
    'person': (0, 165, 255),     # Orange
    'bus': (255, 200, 0),        # Sky Blue
    'bicycle': (0, 255, 255)     # Yellow
}

# Zone colors for drawing
ZONE_COLORS = [
    (0, 100, 255),   # Orange-Red
    (255, 0, 200),   # Magenta
    (0, 255, 0),     # Green
    (255, 200, 0),   # Cyan
    (255, 0, 100)    # Purple
]

# ==================== INTERACTIVE ZONE DRAWER ====================
class ZoneDrawer:
    def __init__(self, frame, video_path):
        self.original = frame.copy()
        self.zones = []
        self.current_zone = []
        self.drawing = False
        self.zone_names = []
        self.video_path = video_path
        
        # Calculate display size to fit screen (max 1600x900)
        self.orig_height, self.orig_width = frame.shape[:2]
        max_width, max_height = 1600, 900
        
        # Calculate scaling factor
        scale_w = max_width / self.orig_width
        scale_h = max_height / self.orig_height
        self.scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
        
        # Calculate display dimensions
        self.display_width = int(self.orig_width * self.scale)
        self.display_height = int(self.orig_height * self.scale)
        
        # Resize frame for display
        self.frame = cv2.resize(self.original, (self.display_width, self.display_height))
        
        print(f"\nOriginal video: {self.orig_width}x{self.orig_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        print(f"Scale factor: {self.scale:.2f}")
        
        # Window setup
        self.window_name = "ZONE DRAWER - Draw zones and press ENTER"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to original coordinates
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            
            # Add point to current zone (in original coordinates)
            self.current_zone.append((orig_x, orig_y))
            self.drawing = True
            self.redraw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current zone
            if len(self.current_zone) >= 3:
                zone_num = len(self.zones) + 1
                zone_name = f"ZONE {zone_num}"
                self.zones.append({
                    'name': zone_name,
                    'points': self.current_zone.copy(),
                    'color': ZONE_COLORS[len(self.zones) % len(ZONE_COLORS)]
                })
                print(f"  ✓ Zone {zone_num} created with {len(self.current_zone)} points")
                self.current_zone = []
                self.drawing = False
                self.redraw()
    
    def redraw(self):
        # Start with resized original
        self.frame = cv2.resize(self.original, (self.display_width, self.display_height))
        
        # Optional: Draw grid lines for better alignment
        grid_color = (40, 40, 40)
        grid_spacing = 50  # pixels
        
        # Vertical grid lines
        for x in range(0, self.display_width, grid_spacing):
            cv2.line(self.frame, (x, 0), (x, self.display_height), grid_color, 1)
        
        # Horizontal grid lines  
        for y in range(0, self.display_height, grid_spacing):
            cv2.line(self.frame, (0, y), (self.display_width, y), grid_color, 1)
        
        # Draw completed zones (scale points for display)
        for i, zone in enumerate(self.zones):
            # Scale points for display
            display_points = np.array([(int(p[0] * self.scale), int(p[1] * self.scale)) 
                                       for p in zone['points']])
            color = zone['color']
            
            # Fill zone with transparency
            overlay = self.frame.copy()
            cv2.fillPoly(overlay, [display_points], color)
            cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
            
            # Draw thicker border for completed zones
            cv2.polylines(self.frame, [display_points], True, (255, 255, 255), 5, lineType=cv2.LINE_AA)
            cv2.polylines(self.frame, [display_points], True, color, 3, lineType=cv2.LINE_AA)
            
            # Draw corner points
            for point in display_points:
                cv2.circle(self.frame, tuple(point), 6, (255, 255, 255), -1)
                cv2.circle(self.frame, tuple(point), 4, color, -1)
            
            # Draw zone label
            centroid_x = int(np.mean([p[0] for p in display_points]))
            centroid_y = int(np.mean([p[1] for p in display_points]))
            
            text = zone['name']
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            
            # Label background
            cv2.rectangle(self.frame,
                         (centroid_x - text_size[0]//2 - 15, centroid_y - text_size[1]//2 - 12),
                         (centroid_x + text_size[0]//2 + 15, centroid_y + text_size[1]//2 + 12),
                         (0, 0, 0), -1)
            cv2.rectangle(self.frame,
                         (centroid_x - text_size[0]//2 - 15, centroid_y - text_size[1]//2 - 12),
                         (centroid_x + text_size[0]//2 + 15, centroid_y + text_size[1]//2 + 12),
                         color, 3)
            cv2.putText(self.frame, text,
                       (centroid_x - text_size[0]//2, centroid_y + text_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Draw current zone being created (scale points for display)
        if len(self.current_zone) > 0:
            for i, point in enumerate(self.current_zone):
                display_point = (int(point[0] * self.scale), int(point[1] * self.scale))
                
                # Draw point with glow effect
                cv2.circle(self.frame, display_point, 10, (0, 255, 255), 2)
                cv2.circle(self.frame, display_point, 6, (0, 255, 255), -1)
                
                # Draw point number
                cv2.putText(self.frame, str(i+1), (display_point[0] + 12, display_point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                
                # Draw lines connecting points
                if i > 0:
                    prev_display = (int(self.current_zone[i-1][0] * self.scale), 
                                  int(self.current_zone[i-1][1] * self.scale))
                    cv2.line(self.frame, prev_display, display_point, (0, 255, 255), 3)
            
            # Draw closing line preview (dashed)
            if len(self.current_zone) >= 2:
                first_display = (int(self.current_zone[0][0] * self.scale), 
                               int(self.current_zone[0][1] * self.scale))
                last_display = (int(self.current_zone[-1][0] * self.scale), 
                              int(self.current_zone[-1][1] * self.scale))
                
                # Dashed line preview
                dx = last_display[0] - first_display[0]
                dy = last_display[1] - first_display[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist > 0:
                    steps = int(dist / 10)
                    for j in range(0, steps, 2):
                        t1 = j / steps
                        t2 = min((j + 1) / steps, 1.0)
                        p1 = (int(first_display[0] + dx * t1), int(first_display[1] + dy * t1))
                        p2 = (int(first_display[0] + dx * t2), int(first_display[1] + dy * t2))
                        cv2.line(self.frame, p1, p2, (100, 255, 255), 2)
            
            # Show instruction at top
            instruction_bg = self.frame.copy()
            cv2.rectangle(instruction_bg, (0, 0), (self.display_width, 70), (0, 0, 0), -1)
            cv2.addWeighted(instruction_bg, 0.7, self.frame, 0.3, 0, self.frame)
            
            cv2.putText(self.frame, f"Drawing Zone {len(self.zones) + 1}  |  Points: {len(self.current_zone)}  |  Right-click to finish (min 3 points)",
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Draw instructions panel (more compact and attractive)
        panel_height = 140
        panel_y = self.display_height - panel_height
        
        # Attractive gradient background
        overlay = self.frame.copy()
        for i in range(panel_height):
            alpha_val = i / panel_height
            color_val = int(10 + 20 * alpha_val)
            cv2.line(overlay, (0, panel_y + i), (self.display_width, panel_y + i),
                    (color_val, color_val, color_val), 1)
        
        # Border
        cv2.line(overlay, (0, panel_y), (self.display_width, panel_y), (0, 200, 255), 3)
        cv2.addWeighted(overlay, 0.8, self.frame, 0.2, 0, self.frame)
        
        # Instructions
        instructions = [
            ("LEFT CLICK", "Add point to zone"),
            ("RIGHT CLICK", "Finish current zone (min 3 points)"),
            ("ENTER", "Start video processing"),
            ("C", "Clear all zones"),
        ]
        
        y_offset = panel_y + 30
        for key, desc in instructions:
            # Key in colored box
            key_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(self.frame, (20, y_offset - 18), (25 + key_size[0], y_offset + 5),
                         (0, 200, 255), 2)
            cv2.putText(self.frame, key, (22, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            
            # Description
            cv2.putText(self.frame, desc, (35 + key_size[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            y_offset += 27
        
        # Status
        status_text = f"Zones Created: {len(self.zones)}"
        cv2.putText(self.frame, status_text, (self.display_width - 200, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        cv2.imshow(self.window_name, self.frame)
    
    def run(self):
        print("\n" + "=" * 70)
        print("INTERACTIVE ZONE DRAWING")
        print("=" * 70)
        print("Instructions:")
        print("  1. LEFT CLICK to add points to create a zone")
        print("  2. RIGHT CLICK to finish the current zone")
        print("  3. Repeat for multiple zones")
        print("  4. Press ENTER when done to start processing")
        print("=" * 70)
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter key
                if len(self.zones) > 0:
                    cv2.destroyWindow(self.window_name)
                    return self.zones
                else:
                    print("  Please create at least one zone!")
            
            elif key == 27:  # ESC key
                cv2.destroyWindow(self.window_name)
                return None
            
            elif key == ord('c'):  # Clear all zones
                self.zones = []
                self.current_zone = []
                print("  All zones cleared")
                self.redraw()

# ==================== HELPER FUNCTIONS ====================
def point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def draw_gradient_line(img, pt1, pt2, color1, color2, thickness=2):
    """Draw gradient line"""
    x1, y1 = pt1
    x2, y2 = pt2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    steps = int(distance)
    
    if steps == 0:
        return
    
    for i in range(steps):
        alpha = i / steps
        x = int(x1 + (x2 - x1) * alpha)
        y = int(y1 + (y2 - y1) * alpha)
        
        b = int(color1[0] + (color2[0] - color1[0]) * alpha)
        g = int(color1[1] + (color2[1] - color1[1]) * alpha)
        r = int(color1[2] + (color2[2] - color1[2]) * alpha)
        
        cv2.circle(img, (x, y), thickness, (b, g, r), -1)

def draw_zone_overlay(img, zone, frame_num):
    """Draw monitoring zone"""
    points = np.array(zone['points'])
    color = zone['color']
    
    # Transparent fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    alpha = 0.25 + 0.1 * math.sin(frame_num * 0.05)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Animated border
    thickness = int(3 + 2 * abs(math.sin(frame_num * 0.1)))
    cv2.polylines(img, [points], True, (255, 255, 255), thickness + 2, lineType=cv2.LINE_AA)
    cv2.polylines(img, [points], True, color, thickness, lineType=cv2.LINE_AA)
    
    # Zone label
    centroid_x = int(np.mean([p[0] for p in zone['points']]))
    centroid_y = int(np.mean([p[1] for p in zone['points']]))
    
    text = zone['name']
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    cv2.rectangle(img,
                 (centroid_x - text_size[0]//2 - 8, centroid_y - text_size[1]//2 - 8),
                 (centroid_x + text_size[0]//2 + 8, centroid_y + text_size[1]//2 + 8),
                 (0, 0, 0), -1)
    cv2.rectangle(img,
                 (centroid_x - text_size[0]//2 - 8, centroid_y - text_size[1]//2 - 8),
                 (centroid_x + text_size[0]//2 + 8, centroid_y + text_size[1]//2 + 8),
                 color, 2)
    cv2.putText(img, text,
               (centroid_x - text_size[0]//2, centroid_y + text_size[1]//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

# Dashboard removed - user requested clean output without counting panel

# ==================== MAIN PROCESSING ====================
def process_video(zones, video_path):
    """Process video with detection and counting"""
    print("\n" + "=" * 70)
    print("STARTING VIDEO PROCESSING")
    print("=" * 70)
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)
    names = model.model.names
    
    # Open video
    print(f"Opening video: {video_path}")
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps == 0 or fps is None:
        fps = 25
    
    # Video writer
    output_path = os.path.join(OUTPUT_DIR, "output_with_zones.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Zones: {len(zones)}")
    print("=" * 70)
    
    # Processing variables
    total_counts = defaultdict(int)
    zone_counts = {zone['name']: defaultdict(int) for zone in zones}
    frame_count = 0
    current_fps = 0
    fps_counter = 0
    start_time = time.time()
    last_fps_time = start_time
    
    # Center point for vision lines
    center_point = (width // 2, height // 2)
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            last_fps_time = current_time
        
        # Object detection
        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Draw zones
        for zone in zones:
            draw_zone_overlay(frame, zone, frame_count)
        
        # Draw center point
        pulse_radius = int(8 + 4 * abs(math.sin(frame_count * 0.1)))
        cv2.circle(frame, center_point, pulse_radius, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center_point, 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        
        # Reset frame counts
        frame_counts = defaultdict(int)
        frame_zone_counts = {zone['name']: defaultdict(int) for zone in zones}
        
        # Process detections
        for box, cls in zip(boxes, classes):
            obj_class = names[int(cls)]
            
            # Filter relevant objects
            if obj_class not in COLORS:
                continue
            
            # Get box center
            x1, y1, x2, y2 = map(int, box)
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Check which zone
            in_zone = False
            for zone in zones:
                if point_in_polygon(box_center, zone['points']):
                    frame_counts[obj_class] += 1
                    frame_zone_counts[zone['name']][obj_class] += 1
                    in_zone = True
                    break
            
            if not in_zone:
                continue
            
            # Get color
            color = COLORS[obj_class]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Draw label
            label = obj_class.upper()
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 6),
                         (x1 + label_size[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
            # Draw vision line
            draw_gradient_line(frame, center_point, box_center, (255, 255, 0), color, 2)
            
            # Draw node
            cv2.circle(frame, box_center, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, box_center, 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Update total counts
        total_counts = frame_counts.copy()
        
        # NO DASHBOARD - User requested removal
        
        # Frame counter (small, bottom left)
        cv2.putText(frame, f"Frame: {frame_count:05d} | FPS: {current_fps:02d} | Total: {sum(total_counts.values()):04d}",
                   (15, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        # Write and display
        writer.write(frame)
        cv2.imshow("Processing Video - Press Q to stop", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        # Progress with detailed counts in console
        if frame_count % 50 == 0:
            cars = total_counts.get('car', 0)
            trucks = total_counts.get('truck', 0)
            bikes = total_counts.get('motorcycle', 0)
            people = total_counts.get('person', 0)
            total = sum(total_counts.values())
            
            print(f"Frame {frame_count:05d} | FPS: {current_fps:02d} | "
                  f"Cars: {cars:03d} | Trucks: {trucks:03d} | Bikes: {bikes:03d} | People: {people:03d} | Total: {total:04d}")
    
    # Cleanup
    cam.release()
    writer.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Time: {total_time:.2f}s")
    print(f"Avg FPS: {frame_count/total_time:.2f}")
    print("=" * 70)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTERACTIVE TRAFFIC MONITORING SYSTEM")
    print("=" * 70)
    
    # Open video to get first frame
    cam = cv2.VideoCapture(VIDEO_INPUT)
    if not cam.isOpened():
        print(f"ERROR: Cannot open video file: {VIDEO_INPUT}")
        exit(1)
    
    success, first_frame = cam.read()
    cam.release()
    
    if not success:
        print("ERROR: Cannot read video frame")
        exit(1)
    
    # Interactive zone drawing
    drawer = ZoneDrawer(first_frame, VIDEO_INPUT)
    zones = drawer.run()
    
    if zones is None or len(zones) == 0:
        print("\nNo zones created. Exiting...")
        exit(0)
    
    print(f"\nZones created: {len(zones)}")
    for i, zone in enumerate(zones, 1):
        print(f"  Zone {i}: {len(zone['points'])} points")
    
    # Save zones to file
    zones_file = os.path.join(OUTPUT_DIR, "zones_config.json")
    with open(zones_file, 'w') as f:
        json.dump(zones, f, indent=2)
    print(f"\nZones saved to: {zones_file}")
    
    # Start processing
    input("\nPress ENTER to start processing video...")
    process_video(zones, VIDEO_INPUT)
import cv2
import numpy as np
import re
import easyocr
import math
import time
from collections import defaultdict

reader = None
LH_ABOVE_PRESSURE_DISTANCE = 20  # L/H is exactly 20px above pressure

def get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def detect_lh_markers(img, header_mask=None):
    start_time = time.time()
    reader = get_ocr_reader()
    height, width = img.shape[:2]
    
    # Basic image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Apply header mask if provided
    masked_thresh = cv2.bitwise_and(thresh, thresh, mask=header_mask) if header_mask is not None else thresh
    
    # Create debug image
    debug_img = img.copy()
    
    # Initialize result containers
    l_markers = []
    h_markers = []
    pressure_values = []
    
    # Step 1: Find all pressure values (3-4 digit numbers around 1000)
    print("\nStep 1: Detecting pressure values...")
    try:
        num_results = reader.readtext(img, allowlist='0123456789', text_threshold=0.3)
        
        for (bbox, text, prob) in num_results:
            if prob < 0.3 or not re.match(r'^\d{3,4}$', text):
                continue
            
            value = int(text)
            if 950 <= value <= 1050:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)
                
                # Skip if in masked header area
                if header_mask is not None and center_y < height * 0.11 and center_x < width * 0.38:
                    continue
                    
                # Create rectangle points for drawing
                rect_points = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                rect_points = rect_points.reshape((-1, 1, 2))
                
                pressure_values.append({
                    'value': value,
                    'position': (center_x, center_y),
                    'bbox': bbox,
                    'rect_points': rect_points,
                    'prob': prob,
                    'assigned': False
                })
                
                # Draw orange rectangle around pressure value
                cv2.polylines(debug_img, [rect_points], True, (0, 165, 255), 2)
                
                # Print detected value
                print(f"Detected pressure value {value} at position ({center_x}, {center_y})")
    except Exception as e:
        print(f"Error detecting pressure values: {e}")
    
    # Step 2: For each pressure value, meticulously scan for L/H markers in 20x20 square above
    print("\nStep 2: Scanning for L/H markers above each pressure value...")
    for pressure in pressure_values:
        px, py = pressure['position']
        
        # Define search area (20x20 square centered 20px above pressure)
        search_center_x = px
        search_center_y = py - LH_ABOVE_PRESSURE_DISTANCE
        
        search_size = 20  # Size of search box
        x1 = max(0, search_center_x - search_size//2)
        y1 = max(0, search_center_y - search_size//2)
        x2 = min(width, search_center_x + search_size//2)
        y2 = min(height, search_center_y + search_size//2)
        
        # Mark search area in debug image with blue rectangle
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        marker_rect_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32).reshape((-1, 1, 2))
        
        # Extract ROI for detailed search
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        # Prepare the ROI for better detection
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY)
        
        # Try multiple approaches to find L/H marker
        found_marker = None
        
        # 1. First try OCR with very low threshold
        try:
            results = reader.readtext(roi, allowlist='LH', text_threshold=0.1, paragraph=False)
            
            # Find the most confident result
            best_prob = 0.1  # Minimum threshold
            best_result = None
            
            for (bbox, text, prob) in results:
                if prob > best_prob and text in ['L', 'H']:
                    best_prob = prob
                    best_result = (bbox, text, prob)
            
            # If we found a result, use it
            if best_result:
                bbox, text, prob = best_result
                (top_left, top_right, bottom_right, bottom_left) = bbox
                center_x = int(x1 + (top_left[0] + bottom_right[0]) / 2)
                center_y = int(y1 + (top_left[1] + bottom_right[1]) / 2)
                
                found_marker = {
                    'position': (center_x, center_y),
                    'text': text.upper(),
                    'bbox': [(x1 + p[0], y1 + p[1]) for p in bbox],
                    'prob': prob,
                    'value': pressure['value'],
                    'marker_rect_points': marker_rect_points,
                    'pressure_position': pressure['position'],
                    'pressure_rect_points': pressure['rect_points']
                }
                print(f"Found {text} marker at {found_marker['position']} above pressure {pressure['value']} (prob: {prob:.2f})")
        except Exception as e:
            print(f"OCR error in search area for pressure {pressure['value']}: {e}")
        
        # 2. If OCR failed, analyze the binary pattern in the region
        if not found_marker:
            # Count white and black pixels in different regions
            left_half = roi_thresh[:, :roi_thresh.shape[1]//2]
            right_half = roi_thresh[:, roi_thresh.shape[1]//2:]
            
            left_white = cv2.countNonZero(left_half)
            right_white = cv2.countNonZero(right_half)
            
            # L has more white pixels on the left and bottom
            # H has more balanced white pixels and vertical strokes
            
            # Simple heuristic: if left half has significantly more white pixels, it's likely L
            text = 'L' if left_white > right_white * 1.5 else 'H'
            
            # Use analysis result
            found_marker = {
                'position': (search_center_x, search_center_y),  # Center of search area
                'text': text,
                'prob': 0.5,  # Medium confidence
                'value': pressure['value'],
                'marker_rect_points': marker_rect_points,
                'pressure_position': pressure['position'],
                'pressure_rect_points': pressure['rect_points'],
                'pattern_analyzed': True
            }
            print(f"Pattern analysis suggests {text} above pressure {pressure['value']} (L:R white pixel ratio = {left_white/max(1, right_white):.2f})")
        
        # Add the marker to the appropriate list
        if found_marker['text'] == 'L':
            l_markers.append(found_marker)
            # Draw green L marker with yellow square
            cv2.circle(debug_img, found_marker['position'], 12, (0, 255, 0), 2)
            cv2.rectangle(debug_img, 
                         (found_marker['position'][0] - 10, found_marker['position'][1] - 10),
                         (found_marker['position'][0] + 10, found_marker['position'][1] + 10),
                         (0, 255, 255), 2)
            cv2.putText(debug_img, "L", (found_marker['position'][0] - 5, found_marker['position'][1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            h_markers.append(found_marker)
            # Draw red H marker with yellow square
            cv2.circle(debug_img, found_marker['position'], 12, (0, 0, 255), 2)
            cv2.rectangle(debug_img, 
                         (found_marker['position'][0] - 10, found_marker['position'][1] - 10),
                         (found_marker['position'][0] + 10, found_marker['position'][1] + 10),
                         (0, 255, 255), 2)
            cv2.putText(debug_img, "H", (found_marker['position'][0] - 5, found_marker['position'][1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw connection to pressure
        color = (0, 255, 0) if found_marker['text'] == 'L' else (0, 0, 255)
        cv2.line(debug_img, found_marker['position'], pressure['position'], color, 2)
        pressure['assigned'] = True
            
    # Summary of detected systems
    print("\nSummary of detected L/H systems:")
    for i, marker in enumerate(l_markers):
        confidence = "pattern analysis" if marker.get('pattern_analyzed', False) else f"OCR prob: {marker['prob']:.2f}"
        print(f"L system {i+1}: L at {marker['position']}, pressure {marker['value']} at {marker['pressure_position']} ({confidence})")
    
    for i, marker in enumerate(h_markers):
        confidence = "pattern analysis" if marker.get('pattern_analyzed', False) else f"OCR prob: {marker['prob']:.2f}"
        print(f"H system {i+1}: H at {marker['position']}, pressure {marker['value']} at {marker['pressure_position']} ({confidence})")
    
    print(f"\nFinal count: {len(l_markers)} L markers, {len(h_markers)} H markers")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    
    return l_markers, h_markers, masked_thresh, debug_img

def format_output_data(l_markers, h_markers, filename):
    timestamp_info = {'date': None, 'time': None, 'raw_text': []}
    
    if filename:
        parts = filename.split('_')
        if len(parts) >= 5:
            time_val = parts[0]
            day_of_week = parts[2] if len(parts) > 2 else ''
            day = parts[3] if len(parts) > 3 else ''
            month = parts[4] if len(parts) > 4 else ''
            timestamp_info['time'] = time_val
            timestamp_info['date'] = f"{day_of_week} {day} {month} {time_val} UTC"
    
    output_data = {
        'timestamp': timestamp_info,
        'l_systems': [],
        'h_systems': [],
        'detection_parameters': {
            'lh_above_pressure_distance': LH_ABOVE_PRESSURE_DISTANCE
        }
    }
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.intc, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        elif isinstance(obj, tuple) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in obj):
            return tuple(int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x for x in obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Process L markers
    for l_marker in l_markers:
        center = l_marker['position']
        if isinstance(center, tuple) and len(center) == 2:
            center = (int(center[0]), int(center[1]))
        
        pressure_position = l_marker.get('pressure_position')
        if pressure_position and isinstance(pressure_position, tuple) and len(pressure_position) == 2:
            pressure_position = (int(pressure_position[0]), int(pressure_position[1]))
        
        l_system = {
            'center': center,
            'pressure': l_marker['value'],
            'text': l_marker['text'],
            'pressure_position': pressure_position,
            'pattern_analyzed': l_marker.get('pattern_analyzed', False),
            'prob': float(l_marker.get('prob', 0.5)),
            'x_points': []  # Empty list since we're not handling X markers
        }
        output_data['l_systems'].append(convert_to_serializable(l_system))
    
    # Process H markers
    for h_marker in h_markers:
        center = h_marker['position']
        if isinstance(center, tuple) and len(center) == 2:
            center = (int(center[0]), int(center[1]))
        
        pressure_position = h_marker.get('pressure_position')
        if pressure_position and isinstance(pressure_position, tuple) and len(pressure_position) == 2:
            pressure_position = (int(pressure_position[0]), int(pressure_position[1]))
        
        h_system = {
            'center': center,
            'pressure': h_marker['value'],
            'text': h_marker['text'],
            'pressure_position': pressure_position,
            'pattern_analyzed': h_marker.get('pattern_analyzed', False),
            'prob': float(h_marker.get('prob', 0.5)),
            'x_points': []  # Empty list since we're not handling X markers
        }
        output_data['h_systems'].append(convert_to_serializable(h_system))
    
    return output_data
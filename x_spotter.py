import cv2
import numpy as np

def detect_x_markers(img, white_mask, existing_x_positions=None):
    if existing_x_positions is None:
        existing_x_positions = []
        
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(white_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_markers = []
    
    x_debug_img = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5 or area > 100:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w//2
        center_y = y + h//2
        
        is_duplicate = False
        for ex_x, ex_y in existing_x_positions:
            if ((center_x - ex_x)**2 + (center_y - ex_y)**2) < 100:
                is_duplicate = True
                break
        if is_duplicate:
            continue
            
        height, width = white_mask.shape
        border_margin = 10
        if (center_x < border_margin or center_x > width - border_margin or 
            center_y < border_margin or center_y > height - border_margin):
            continue
        
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.5 <= aspect_ratio <= 2.0:
            roi = white_mask[y:y+h, x:x+w]
            white_ratio = cv2.countNonZero(roi) / (w * h) if (w * h) > 0 else 0
            
            if 0.2 <= white_ratio <= 0.8:
                x_markers.append({
                    'position': (center_x, center_y),
                    'associated_to': None,
                    'is_l': None,
                    'text': 'X (shape)',
                    'bbox': (x, y, w, h)
                })
                
                cv2.rectangle(x_debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.circle(x_debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
    
    return x_markers

def detect_small_connected_components(white_mask, existing_x_positions=None):
    if existing_x_positions is None:
        existing_x_positions = []
    
    sensitive_mask = white_mask.copy()
    sensitive_mask = cv2.GaussianBlur(sensitive_mask, (3, 3), 0)
    _, sensitive_mask = cv2.threshold(sensitive_mask, 100, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    sensitive_labels, sensitive_stats, sensitive_centroids = None, None, None
    try:
        sensitive_num_labels, sensitive_labels, sensitive_stats, sensitive_centroids = cv2.connectedComponentsWithStats(sensitive_mask, connectivity=8)
    except:
        sensitive_num_labels, sensitive_labels, sensitive_stats, sensitive_centroids = num_labels, labels, stats, centroids
    
    cc_debug_img = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    sensitive_debug_img = cv2.cvtColor(sensitive_mask, cv2.COLOR_GRAY2BGR)
    cc_x_markers = []
    
    def process_components(num_components, component_stats, component_centroids, is_sensitive=False):
        markers_found = []
        for i in range(1, num_components):
            area = component_stats[i, cv2.CC_STAT_AREA]
            min_area = 2 if is_sensitive else 5
            max_area = 150 if is_sensitive else 100
            if area < min_area or area > max_area:
                continue
                
            center_x = int(component_centroids[i][0])
            center_y = int(component_centroids[i][1])
            x = component_stats[i, cv2.CC_STAT_LEFT]
            y = component_stats[i, cv2.CC_STAT_TOP]
            w = component_stats[i, cv2.CC_STAT_WIDTH]
            h = component_stats[i, cv2.CC_STAT_HEIGHT]
            
            aspect_ratio = w / h if h > 0 else 0
            min_ratio = 0.3 if is_sensitive else 0.5
            max_ratio = 3.0 if is_sensitive else 2.0
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                continue
                
            height, width = white_mask.shape
            border_margin = 5 if is_sensitive else 10
            if (center_x < border_margin or center_x > width - border_margin or 
                center_y < border_margin or center_y > height - border_margin):
                continue
            
            is_duplicate = False
            duplicate_threshold = 200 if is_sensitive else 100
            for ex_x, ex_y in existing_x_positions + [(m['position'][0], m['position'][1]) for m in markers_found]:
                if ((center_x - ex_x)**2 + (center_y - ex_y)**2) < duplicate_threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
                
            marker_type = 'X (Sensitive)' if is_sensitive else 'X (CC)'
            color = (0, 255, 255) if is_sensitive else (255, 0, 255)
            
            if area < 10 and is_sensitive:
                nearby_radius = 15
                roi_x1 = max(0, center_x - nearby_radius)
                roi_x2 = min(width, center_x + nearby_radius)
                roi_y1 = max(0, center_y - nearby_radius)
                roi_y2 = min(height, center_y + nearby_radius)
                
                roi = white_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                if np.sum(roi) > 0:
                    marker_type = 'X (Distorted)'
                    color = (0, 160, 255)
            
            markers_found.append({
                'position': (center_x, center_y),
                'associated_to': None,
                'is_l': None,
                'text': marker_type,
                'bbox': (x, y, w, h)
            })
            
            debug_img = sensitive_debug_img if is_sensitive else cc_debug_img
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 1)
            cv2.circle(debug_img, (center_x, center_y), 4, color, -1)
        
        return markers_found
    
    cc_x_markers.extend(process_components(num_labels, stats, centroids))
    
    if sensitive_labels is not None:
        cc_x_markers.extend(process_components(sensitive_num_labels, sensitive_stats, sensitive_centroids, True))
    
    return cc_x_markers

def find_isolated_x_markers(sensitive_mask, existing_x_positions=None):
    if existing_x_positions is None:
        existing_x_positions = []
        
    isolated_x_markers = []
    height, width = sensitive_mask.shape
    
    isolated_debug_img = cv2.cvtColor(sensitive_mask, cv2.COLOR_GRAY2BGR)
    
    header_height = int(height * 0.12)
    header_width = int(width * 0.40)
    
    for y in range(0, height, 5):
        for x in range(0, width, 5):
            if y < header_height and x < header_width:
                continue
                
            region_size = 10
            x1 = max(0, x - region_size//2)
            x2 = min(width, x + region_size//2)
            y1 = max(0, y - region_size//2)
            y2 = min(height, y + region_size//2)
            
            region = sensitive_mask[y1:y2, x1:x2]
            if np.sum(region) > 0:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                is_duplicate = False
                for ex_x, ex_y in existing_x_positions:
                    if ((center_x - ex_x)**2 + (center_y - ex_y)**2) < 400:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    isolated_x_markers.append({
                        'position': (center_x, center_y),
                        'associated_to': None,
                        'is_l': None,
                        'text': 'X (Isolated)',
                        'bbox': (x1, y1, x2-x1, y2-y1)
                    })
                    
                    cv2.rectangle(isolated_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.circle(isolated_debug_img, (center_x, center_y), 3, (0, 0, 255), -1)
                    
                    existing_x_positions.append((center_x, center_y))
    
    return isolated_x_markers
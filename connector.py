import cv2
import numpy as np
import math

MAX_X_TO_LH_DISTANCE = 100

def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def is_point_inside_box(point, box):
    x, y = point
    if isinstance(box, tuple) and len(box) == 4:
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2
    
    if isinstance(box, list):
        contour = np.array(box).reshape((-1, 2))
        return cv2.pointPolygonTest(contour, (x, y), False) >= 0
    
    return False


def consolidate_x_markers(x_markers, max_distance=10):
    """
    Consolidate X markers that are very close to each other
    
    Parameters:
        x_markers (list): List of X markers
        max_distance (float): Maximum distance to consider markers as the same
    
    Returns:
        list: Consolidated list of unique X markers
    """
    if not x_markers:
        return []
    
    # Sort markers by x position to make consolidation more efficient
    sorted_markers = sorted(x_markers, key=lambda x: x['position'][0])
    
    consolidated_markers = []
    current_group = [sorted_markers[0]]
    
    for marker in sorted_markers[1:]:
        # Check if this marker is close to the current group
        is_close = False
        for group_marker in current_group:
            if calculate_distance(marker['position'], group_marker['position']) <= max_distance:
                is_close = True
                break
        
        if is_close:
            current_group.append(marker)
        else:
            # Consolidate the current group to its center
            if current_group:
                # Calculate average position
                avg_x = int(sum(m['position'][0] for m in current_group) / len(current_group))
                avg_y = int(sum(m['position'][1] for m in current_group) / len(current_group))
                
                # Take the first marker as the base and update its position
                consolidated_marker = current_group[0].copy()
                consolidated_marker['position'] = (avg_x, avg_y)
                consolidated_marker['text'] = current_group[0]['text']
                
                consolidated_markers.append(consolidated_marker)
            
            # Start a new group
            current_group = [marker]
    
    # Handle the last group
    if current_group:
        # Calculate average position
        avg_x = int(sum(m['position'][0] for m in current_group) / len(current_group))
        avg_y = int(sum(m['position'][1] for m in current_group) / len(current_group))
        
        # Take the first marker as the base and update its position
        consolidated_marker = current_group[0].copy()
        consolidated_marker['position'] = (avg_x, avg_y)
        consolidated_marker['text'] = current_group[0]['text']
        
        consolidated_markers.append(consolidated_marker)
    
    return consolidated_markers

def get_valid_x_markers(x_markers, l_markers, h_markers):
    # First, consolidate x markers
    consolidated_x_markers = consolidate_x_markers(x_markers)
    boxes = []
    
    # Add pressure and marker rectangular points
    for marker in l_markers + h_markers:
        # Pressure rectangular points box
        if 'pressure_rect_points' in marker:
            rect_points = marker['pressure_rect_points']
            x_coords = [point[0][0] for point in rect_points]
            y_coords = [point[0][1] for point in rect_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            boxes.append((x_min-3, y_min-3, x_max+3, y_max+3))
        
        # Marker rectangular points box
        if 'marker_rect_points' in marker:
            rect_points = marker['marker_rect_points']
            x_coords = [point[0][0] for point in rect_points]
            y_coords = [point[0][1] for point in rect_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            boxes.append((x_min-3, y_min-3, x_max+3, y_max+3))
    
    # Filter valid X markers
    valid_x_markers = []
    for x_marker in consolidated_x_markers:
        x_pos = x_marker['position']
        
        is_valid = True
        for box in boxes:
            if is_point_inside_box(x_pos, box):
                is_valid = False
                break
        
        if is_valid:
            valid_x_markers.append(x_marker)
    
    return valid_x_markers

def connect_x_markers_to_lh(x_markers, l_markers, h_markers, debug_img=None):
    valid_x_markers = get_valid_x_markers(x_markers, l_markers, h_markers)
    
    l_markers_updated = [dict(marker) for marker in l_markers]
    h_markers_updated = [dict(marker) for marker in h_markers]
    connected_x_markers = []
    
    # Dodajemy pole "associated_x" do każdego markera L i H
    for marker in l_markers_updated + h_markers_updated:
        marker['associated_x'] = []
        marker['assigned'] = False  # Flaga, czy już przypisano x_marker
    
    # Tworzymy listę z możliwymi połączeniami: (x_marker, marker, distance, is_l_system)
    possible_connections = []
    
    for x_marker in valid_x_markers:
        x_pos = x_marker['position']
        
        for l_marker in l_markers_updated:
            l_pos = l_marker['position']
            dist = calculate_distance(x_pos, l_pos)
            if dist <= MAX_X_TO_LH_DISTANCE:
                possible_connections.append((x_marker, l_marker, dist, True))
        
        for h_marker in h_markers_updated:
            h_pos = h_marker['position']
            dist = calculate_distance(x_pos, h_pos)
            if dist <= MAX_X_TO_LH_DISTANCE:
                possible_connections.append((x_marker, h_marker, dist, False))
    
    # Sortujemy połączenia po najmniejszej odległości
    possible_connections.sort(key=lambda item: item[2])
    
    used_x_markers = set()
    
    for x_marker, target_marker, dist, is_l_system in possible_connections:
        x_id = id(x_marker)
        target_id = id(target_marker)
        
        # Jeśli x_marker nie został jeszcze użyty i target nie jest przypisany
        if x_id not in used_x_markers and not target_marker['assigned']:
            connected_x = x_marker.copy()
            connected_x['associated_to'] = target_marker['position']
            connected_x['is_l'] = is_l_system
            connected_x['distance'] = dist
            
            connected_x_markers.append(connected_x)
            target_marker['associated_x'].append(connected_x)
            target_marker['assigned'] = True
            used_x_markers.add(x_id)
            
            if debug_img is not None:
                color = (0, 255, 0) if is_l_system else (0, 0, 255)
                cv2.line(debug_img, x_marker['position'], target_marker['position'], color, 1, cv2.LINE_AA)
    
    return l_markers_updated, h_markers_updated, connected_x_markers


def create_connection_image(img, l_markers, h_markers, x_markers):
    connection_img = img.copy()
    valid_x_markers = get_valid_x_markers(x_markers, l_markers, h_markers)
    
    for l_marker in l_markers:
        center_x, center_y = l_marker['position']
        cv2.circle(connection_img, (center_x, center_y), 15, (0, 255, 0), 2)
        
        for x in l_marker.get('associated_x', []):
            x_pos = x['position']
            cv2.circle(connection_img, x_pos, 5, (255, 0, 0), -1)
            cv2.line(connection_img, l_marker['position'], x_pos, (0, 255, 0), 1, cv2.LINE_AA)
    
    for h_marker in h_markers:
        center_x, center_y = h_marker['position']
        cv2.circle(connection_img, (center_x, center_y), 15, (0, 0, 255), 2)
        
        for x in h_marker.get('associated_x', []):
            x_pos = x['position']
            cv2.circle(connection_img, x_pos, 5, (255, 0, 0), -1)
            cv2.line(connection_img, h_marker['position'], x_pos, (0, 0, 255), 1, cv2.LINE_AA)
    
    for x_marker in valid_x_markers:
        if not any('associated_x' in marker and x_marker in marker['associated_x'] 
                   for marker in l_markers + h_markers):
            x_pos = x_marker['position']
            cv2.circle(connection_img, x_pos, 5, (128, 128, 128), -1)
    
    return connection_img

def update_output_data(output_data, l_markers, h_markers):
    updated_data = output_data.copy()
    
    for i, l_marker in enumerate(l_markers):
        if i < len(updated_data['l_systems']):
            x_points = [x['position'] for x in l_marker.get('associated_x', [])]
            updated_data['l_systems'][i]['x_points'] = x_points
    
    for i, h_marker in enumerate(h_markers):
        if i < len(updated_data['h_systems']):
            x_points = [x['position'] for x in h_marker.get('associated_x', [])]
            updated_data['h_systems'][i]['x_points'] = x_points
    
    updated_data['detection_parameters']['max_x_to_lh_distance'] = MAX_X_TO_LH_DISTANCE
    
    return updated_data

if __name__ == "__main__":
    print("Connector module for X marker association")
    print(f"Maximum X to L/H distance: {MAX_X_TO_LH_DISTANCE} pixels")
import cv2
import numpy as np
import os
import csv
import argparse
from PIL import Image

# Import from the other modules
from x_spotter import detect_x_markers, detect_small_connected_components, find_isolated_x_markers
from LH_spotter import detect_lh_markers, format_output_data
from connector import connect_x_markers_to_lh, create_connection_image, update_output_data

# Define version
__version__ = '1.3.0'

def parse_filename_date(filename):
    """Extract date, month, and time from filename."""
    try:
        # Assuming filename format like 0000_UTC_Wed_03_JAN_mask_distance_debug.jpg
        parts = filename.split('_')
        time = parts[0]
        day = parts[3]
        month = parts[4]
        
        # Convert month abbreviation to number
        month_map = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08', 
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }
        
        return {
            'time': time,
            'day': day,
            'month': month_map.get(month, '00')
        }
    except:
        return {'time': '', 'day': '', 'month': ''}

def detect_weather_elements(image_path, debug=False):
    """
    Main function to detect weather elements in an image
    
    Parameters:
        image_path (str): Path to the image file
        debug (bool): Enable debug output
        
    Returns:
        tuple: (result_img, output_data, debug_img, date_info)
    """
    # Extract the filename for timestamp information
    filename = os.path.basename(image_path)
    mask_name = os.path.splitext(filename)[0]
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise Exception(f"Failed to load image: {e}")
    
    # Create result and debug images
    result_img = img.copy()
    debug_img = img.copy()
    
    # Create header mask - set header area to black in the mask
    height, width = img.shape[:2]
    header_height = int(height * 0.11)
    header_width = int(width * 0.38)
    
    header_mask = np.ones((height, width), dtype=np.uint8) * 255
    header_mask[0:header_height, 0:header_width] = 0
    
    # Apply the header mask to the image to make header area black
    masked_img = cv2.bitwise_and(img, img, mask=header_mask)
    
    # Detect X markers
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    x_markers = []
    existing_x_positions = []
    
    # Detect X markers using shape-based methods
    shape_based_x_markers = detect_x_markers(masked_img, white_mask, existing_x_positions)
    x_markers.extend(shape_based_x_markers)
    existing_x_positions.extend([marker['position'] for marker in shape_based_x_markers])
    
    # Detect X markers using connected components
    cc_based_x_markers = detect_small_connected_components(white_mask, existing_x_positions)
    x_markers.extend(cc_based_x_markers)
    existing_x_positions.extend([marker['position'] for marker in cc_based_x_markers])
    
    # Find isolated X markers
    _, sensitive_mask = cv2.threshold(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    isolated_x_markers = find_isolated_x_markers(sensitive_mask, existing_x_positions)
    x_markers.extend(isolated_x_markers)
    
    # Detect L and H markers
    l_markers, h_markers, white_mask, distance_debug_img = detect_lh_markers(masked_img, header_mask)
    
    # Connect X markers to L/H systems
    connection_img = result_img.copy()
    updated_l_markers, updated_h_markers, _ = connect_x_markers_to_lh(x_markers, l_markers, h_markers, connection_img)
    
    # Format output data
    output_data = format_output_data(l_markers, h_markers, mask_name)
    updated_output_data = update_output_data(output_data, updated_l_markers, updated_h_markers)
    
    # Draw markers and annotations
    for l_marker in updated_l_markers:
        center_x, center_y = l_marker['position']
        cv2.circle(result_img, (center_x, center_y), 15, (0, 255, 0), 2)
        if l_marker.get('value'):
            cv2.putText(result_img, str(l_marker['value']), (center_x + 15, center_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for h_marker in updated_h_markers:
        center_x, center_y = h_marker['position']
        cv2.circle(result_img, (center_x, center_y), 15, (0, 0, 255), 2)
        if h_marker.get('value'):
            cv2.putText(result_img, str(h_marker['value']), (center_x + 15, center_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Extract date info from filename
    date_info = parse_filename_date(filename)
    
    return result_img, updated_output_data, debug_img, date_info, x_markers

def main():
    """Process mask images in a folder."""
    parser = argparse.ArgumentParser(description='Weather System Spotter')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"Weather System Spotter v{__version__}")
    
    masks_folder = 'masks'
    output_folder = 'results'
    debug_folder = 'debug'
    
    # Create necessary folders
    for folder in [output_folder, debug_folder]:
        os.makedirs(folder, exist_ok=True)
        print(f"Created/verified folder: '{folder}'")
    
    if not os.path.exists(masks_folder):
        print(f"Error: Masks folder '{masks_folder}' not found")
        return
    
    mask_files = [f for f in os.listdir(masks_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not mask_files:
        print(f"No image files found in '{masks_folder}' folder")
        return
    
    print(f"Found {len(mask_files)} mask files to process")
    
    # Prepare CSV for output
    csv_path = os.path.join(output_folder, 'weather_systems.csv')
    csv_headers = ['day', 'month', 'hour', 'x', 'y', 'type', 'pressure']
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        
        for i, mask_file in enumerate(mask_files):
            mask_path = os.path.join(masks_folder, mask_file)
            mask_name = os.path.splitext(mask_file)[0]
            
            print(f"\nProcessing mask [{i+1}/{len(mask_files)}]: {mask_file}")
            
            try:
                result_img, output_data, debug_img, date_info, x_markers = detect_weather_elements(mask_path, debug=args.debug)
                
                # Write L and H systems to CSV
                for system_type in ['l_systems', 'h_systems']:
                    for system in output_data.get(system_type, []):
                        # Determine type and associated pressure
                        sys_type = 'L' if system_type == 'l_systems' else 'H'
                        pressure = system.get('pressure', '')
                        
                        # Get associated X markers
                        for x_point in system.get('x_points', []):
                            x, y = x_point
                            
                            csv_writer.writerow([
                                date_info['day'], 
                                date_info['month'], 
                                date_info['time'], 
                                x, 
                                y, 
                                sys_type, 
                                pressure
                            ])
                            
                # Save debug images if debug mode is on
                if args.debug:
                    result_path = os.path.join(output_folder, f"{mask_name}_result.jpg")
                    cv2.imwrite(result_path, result_img)
                    
                    debug_path = os.path.join(debug_folder, f"{mask_name}_debug.jpg")
                    cv2.imwrite(debug_path, debug_img)
                
                print(f"  Processed: {mask_file}")
                
            except Exception as e:
                print(f"  Error processing mask {mask_file}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nProcessing complete! Results saved to {csv_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()
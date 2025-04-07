import cv2
import numpy as np
import os
import re
from PIL import Image
import easyocr

# Global variable for EasyOCR reader to avoid initializing multiple times
ocr_reader = None

def get_ocr_reader():
    """Initialize OCR reader once and reuse it"""
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'])
    return ocr_reader

def extract_timestamp(img):
    """
    Extract date/time information from the top-left corner of weather maps
    
    Args:
        img: Original image
            
    Returns:
        tuple: (full_timestamp, formatted_timestamp, hour)
    """
    # Extract top-left region where timestamp consistently appears
    height, width = img.shape[:2]
    top_left_region = img[0:int(height*0.1), 0:int(width*0.3)]
    
    # Convert to grayscale if not already
    if len(top_left_region.shape) == 3:
        top_left_gray = cv2.cvtColor(top_left_region, cv2.COLOR_BGR2GRAY)
    else:
        top_left_gray = top_left_region
    
    # Use threshold for better text extraction
    _, top_left_binary = cv2.threshold(top_left_gray, 170, 255, cv2.THRESH_BINARY)
    
    # Initialize OCR reader
    reader = get_ocr_reader()
    
    # Extract text using OCR
    date_results = reader.readtext(top_left_binary)
    
    # Look for the specific format with "Valid" keyword
    full_pattern = re.compile(r'Valid\s*(\d{4})\s*UTC\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*(\d{2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)')
    
    for _, text, prob in date_results:
        if prob > 0.2:
            full_match = full_pattern.search(text)
            if full_match:
                time = full_match.group(1)
                day_of_week = full_match.group(2)
                day = full_match.group(3)
                month = full_match.group(4)
                
                # Extract hour
                try:
                    hour = int(time)
                except ValueError:
                    hour = None
                
                full_timestamp = f"{time} UTC {day_of_week} {day} {month}"
                return full_timestamp, full_timestamp.replace(" ", "_"), hour
    
    # Fallback to more general pattern if specific one fails
    general_pattern = re.compile(r'(\d{4})?\s*UTC\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*(\d{2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)')
    
    for _, text, prob in date_results:
        if prob > 0.2:
            general_match = general_pattern.search(text)
            if general_match:
                time = general_match.group(1) or '0000'
                day_of_week = general_match.group(2)
                day = general_match.group(3)
                month = general_match.group(4)
                
                # Extract hour
                try:
                    hour = int(time)
                except ValueError:
                    hour = None
                
                full_timestamp = f"{time} UTC {day_of_week} {day} {month}"
                return full_timestamp, full_timestamp.replace(" ", "_"), hour
    
    return None, None, None

def get_filename_from_timestamp(full_timestamp, base_filename):
    """
    Convert timestamp to valid filename with consistent format
    
    Args:
        full_timestamp: Full timestamp string
        base_filename: Original filename (as fallback)
            
    Returns:
        str: Valid filename
    """
    if not full_timestamp:
        return f"?_UTC_{base_filename}_mask.jpg"
    
    # Use specific timestamp parsing to match exactly the format you want
    timestamp_parts = full_timestamp.split()
    
    # Ensure we have enough parts to construct the filename
    if len(timestamp_parts) >= 4:
        # Construct filename in the format: 0600_UTC_Tue_02_JAN
        filename = f"{timestamp_parts[0]}_UTC_{timestamp_parts[2]}_{timestamp_parts[3]}_{timestamp_parts[4]}"
    else:
        # Fallback if timestamp doesn't match expected format
        filename = full_timestamp.replace(" ", "_")
    
    # Remove any remaining special characters
    filename = re.sub(r'[^\w\-_.]', '', filename)
    
    return f"{filename}_mask.jpg"

def load_image(image_path):
    """
    Load image from file using OpenCV or PIL
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of the image or None if loading fails
    """
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode == 'RGBA' or pil_img.mode == 'P':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None
    return img

def contrast_enhanced(gray):
    """
    Apply contrast enhancement to grayscale image
    
    Args:
        gray: Grayscale image
        
    Returns:
        Binary mask of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, mask = cv2.threshold(enhanced, 220, 255, cv2.THRESH_BINARY)
    return mask

def process_image(image_path, output_folder, base_filename):
    """
    Process a single image: extract timestamp and create mask
    
    Args:
        image_path: Path to input image
        output_folder: Folder to save processed mask
        base_filename: Original filename
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    img = load_image(image_path)
    if img is None:
        return False
    
    # Extract timestamp for naming
    full_timestamp, _, _ = extract_timestamp(img)
    
    # Determine output filename
    output_filename = get_filename_from_timestamp(full_timestamp, base_filename)
    
    if full_timestamp:
        print(f"  - Timestamp detected: {full_timestamp}")
    else:
        print("  - No timestamp detected, using fallback filename")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    try:
        mask = contrast_enhanced(gray)
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, mask)
        print(f"  - Saved as: {output_filename}")
    except Exception as e:
        print(f"  - Error processing image: {e}")
        return False
    
    return True

def process_maps_folder(input_folder, output_folder):
    """
    Process all images in a folder
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save processed masks
    """
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)
    
    # Filter image files
    image_files = [f for f in files if not f.startswith('.') and 
                  f.lower().endswith(('.gif', '.jpg', '.jpeg', '.png'))]
    
    success_count = 0
    fail_count = 0
    
    print(f"Found {len(image_files)} image files in {input_folder}")
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        name_without_ext = os.path.splitext(filename)[0]
        
        print(f"\nProcessing: {filename}")
        
        if process_image(input_path, output_folder, name_without_ext):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {success_count} images")
    print(f"  - Failed to process: {fail_count} images")
    print(f"  - Masks saved to: {output_folder}")

def main():
    """
    Main function to process weather map images
    """
    input_folder = "Maps"
    output_folder = "masks"
    process_maps_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
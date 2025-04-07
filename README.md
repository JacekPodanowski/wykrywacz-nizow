# Weather System Spotter

Weather System Spotter is an advanced Python script for automatic analysis of weather maps, enabling detection and classification of meteorological systems (L and H) along with their characteristics.

# Installation
git clone https://github.com/your_project/weather-system-spotter.git

- Install the required libraries:
- OpenCV (cv2)
- NumPy
- EasyOCR
- Pillow
- Requests
- BeautifulSoup

# Usage

## Generating Masks masker2.py
- Processes maps from the Maps folder
- Generates masks in the masks folder

## Map Analysis main_spotter.py
- Combines all subassemblies
- Analyzes masks from the masks folder
- Generates results in the results folder

## Map Downloading loader.py
- Brutally downloads weather maps from the Met Office repository
- Saves in the download folder
- "I understand the issue now. The script needs to navigate deeper into the site structure."

## Debug Mode: 
- Add the --debug flag to main_spotter.py

![spotter1](https://github.com/user-attachments/assets/f2a460c4-2008-4ad7-b49b-d8a303d367aa)

![spotter 2](https://github.com/user-attachments/assets/f0b3455b-d908-48b9-974d-be4fab29bcaa)



# Author 
Jacek Podanowski feat. Claude 3.7


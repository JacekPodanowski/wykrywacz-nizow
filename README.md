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

## Debug Mode: 
- Add the --debug flag to main_spotter.py

# Author 
Jacek Podanowski feat. Claude 3.7

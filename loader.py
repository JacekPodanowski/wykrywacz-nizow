import requests
from bs4 import BeautifulSoup
import os
import re
import time
import logging
from urllib.parse import urljoin, urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metoffice_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class MetOfficeScraper:
    def __init__(self, base_url, download_dir="download"):
        self.base_url = base_url
        self.download_dir = download_dir
        self.session = requests.Session()
        # Set user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Create the download directory if it doesn't exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logger.info(f"Created download directory: {download_dir}")
    
    def get_soup(self, url):
        """Get BeautifulSoup object from URL with error handling"""
        try:
            logger.info(f"Accessing URL: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing {url}: {e}")
            # Add a small delay before retrying
            time.sleep(2)
            try:
                # Try one more time
                logger.info(f"Retrying URL: {url}")
                response = self.session.get(url)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except requests.exceptions.RequestException as e:
                logger.error(f"Error on retry for {url}: {e}")
                return None
    
    def get_month_folders(self):
        """Get all month folders from the main page"""
        logger.info(f"Getting month folders from: {self.base_url}")
        
        soup = self.get_soup(self.base_url)
        if not soup:
            return []
        
        month_folders = []
        
        # Look for links that might be month folders
        for link in soup.find_all('a', class_="new-primary"):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and text:
                # Skip the main library folder
                if "ASXX Charts - April 2003 onwards" in text:
                    logger.info(f"Skipping main library folder: {text}")
                    continue
                    
                # Check if the text looks like a month folder (format: YYYY_MM_ASXX)
                if re.search(r'^\d{4}_\d{2}', text) or "ASXX" in text:
                    month_folders.append((text, urljoin(self.base_url, href)))
                # Also check for traditional month names
                elif re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', 
                            text.lower()):
                    month_folders.append((text, urljoin(self.base_url, href)))
        
        logger.info(f"Found {len(month_folders)} month folders")
        for name, url in month_folders:
            logger.info(f" - {name}: {url}")
            
        return month_folders
    
    def get_map_links(self, folder_url):
        """Get all map links from a month folder"""
        logger.info(f"Getting map links from: {folder_url}")
        
        soup = self.get_soup(folder_url)
        if not soup:
            return []
        
        map_links = []
        
        # Look for links to individual map pages
        for link in soup.find_all('a', class_=re.compile(r'new-primary')):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and text:
                # Look for any map links - broader criteria
                if ('LIBRARY' in text) or ('FSX' in text) or ('ASX' in text) or re.search(r'\d{4}_\d{2}_\d{2}', text):
                    map_links.append((text, href))
        
        logger.info(f"Found {len(map_links)} map links in {folder_url}")
        if map_links:
            logger.info(f"Sample map links: {map_links[:3]}")
        
        # Check for pagination
        next_page = soup.find('a', class_=re.compile(r'next|pagination-next'))
        if next_page and next_page.get('href'):
            next_url = urljoin(folder_url, next_page.get('href'))
            logger.info(f"Found next page: {next_url}")
            # Get links from next page too
            next_links = self.get_map_links(next_url)
            map_links.extend(next_links)
        
        return map_links
    
    def get_download_url(self, map_page_url):
        """Get the download URL from a map page"""
        logger.info(f"Getting download URL from: {map_page_url}")
        
        soup = self.get_soup(map_page_url)
        if not soup:
            return None
        
        # Look for download button/link
        download_link = soup.find('a', class_=re.compile(r'fa-download'))
        
        if download_link and download_link.get('href'):
            download_url = download_link.get('href')
            logger.info(f"Found download URL: {download_url}")
            return download_url
        
        logger.warning(f"No download URL found at {map_page_url}")
        return None
    
    def download_file(self, url, subfolder=None):
        """Download a file from the given URL"""
        if not url:
            return False
            
        # Create subfolder if needed
        download_path = self.download_dir
        if subfolder:
            download_path = os.path.join(download_path, subfolder)
            if not os.path.exists(download_path):
                os.makedirs(download_path)
        
        # Get filename from URL or Content-Disposition header
        try:
            head_response = self.session.head(url)
            
            # Try to get filename from Content-Disposition header
            filename = None
            if 'Content-Disposition' in head_response.headers:
                content_disp = head_response.headers['Content-Disposition']
                matches = re.findall(r'filename="?([^"]+)"?', content_disp)
                if matches:
                    filename = matches[0]
            
            # If no filename found, use the URL path
            if not filename:
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
            
            # If still no filename, use a timestamp
            if not filename or filename == '':
                filename = f"map_{int(time.time())}.png"
                
            # Clean up filename
            filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
            
            file_path = os.path.join(download_path, filename)
            
            # Skip if file already exists
            if os.path.exists(file_path):
                logger.info(f"File already exists: {file_path}")
                return True
            
            # Download the file
            logger.info(f"Downloading file from {url} to {file_path}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded to: {file_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading from {url}: {e}")
            return False
    
    def download_all_maps(self):
        """Main function to download all maps from the site"""
        logger.info("Starting the download process")
        
        # Get all month folders
        month_folders = self.get_month_folders()
        if not month_folders:
            logger.error("No month folders found, exiting")
            return False
        
        total_downloaded = 0
        
        # Process each month folder
        for month_name, month_url in month_folders:
            logger.info(f"Processing month folder: {month_name}")
            
            # Create a clean subfolder name
            subfolder = re.sub(r'[^\w\-_\. ]', '_', month_name)
            
            # Get all map links in this month folder
            map_links = self.get_map_links(month_url)
            
            if not map_links:
                logger.warning(f"No map links found in {month_name}, checking if this is a deeper structure")
                # This might be a folder with sub-folders, try to explore it
                soup = self.get_soup(month_url)
                if soup:
                    sub_folders = []
                    for link in soup.find_all('a', class_=re.compile(r'new-primary')):
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        if href and text:
                            sub_folders.append((text, urljoin(month_url, href)))
                    
                    if sub_folders:
                        logger.info(f"Found {len(sub_folders)} potential sub-folders in {month_name}")
                        # Process each sub-folder
                        for sub_name, sub_url in sub_folders:
                            sub_folder_name = os.path.join(subfolder, re.sub(r'[^\w\-_\. ]', '_', sub_name))
                            sub_map_links = self.get_map_links(sub_url)
                            
                            # Process each map in the sub-folder
                            for map_name, map_url in sub_map_links:
                                logger.info(f"Processing map from sub-folder: {sub_name} - {map_name}")
                                download_url = self.get_download_url(map_url)
                                if download_url and self.download_file(download_url, sub_folder_name):
                                    total_downloaded += 1
                                time.sleep(1)
            
            # Process each map link in the main folder
            for map_name, map_url in map_links:
                logger.info(f"Processing map: {map_name}")
                
                # Get the download URL
                download_url = self.get_download_url(map_url)
                
                # Download the file
                if download_url and self.download_file(download_url, subfolder):
                    total_downloaded += 1
                
                # Small delay to avoid overwhelming the server
                time.sleep(1)
        
        logger.info(f"Download complete. Total maps downloaded: {total_downloaded}")
        return True

def main():
    # URL of the MetOffice digital archive
    base_url = "https://digital.nmla.metoffice.gov.uk/SO_3c7b44de-30b9-4fec-a385-13326c17aada/"
    
    try:
        scraper = MetOfficeScraper(base_url)
        scraper.download_all_maps()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.exception("Detailed traceback:")

if __name__ == "__main__":
    main()
Weather System Spotter
Opis projektu
Weather System Spotter to zaawansowany skrypt Python do automatycznej analizy map pogodowych, umożliwiający wykrywanie i klasyfikację systemów meteorologicznych (L i H) wraz z ich charakterystyką.
Główne funkcje

Automatyczne rozpoznawanie systemów niskiego (L) i wysokiego (H) ciśnienia
Ekstrakcja wartości ciśnienia
Identyfikacja dodatkowych znaczników X
Generowanie plików CSV z danymi o systemach pogodowych
Tworzenie masek i obrazów debugowych

Wymagania

Python 3.8+
Biblioteki:

OpenCV (cv2)
NumPy
EasyOCR
Pillow
Requests
BeautifulSoup



Instalacja

Sklonuj repozytorium

bashCopygit clone https://github.com/twoj_projekt/weather-system-spotter.git

Zainstaluj wymagane biblioteki

bashCopypip install -r requirements.txt
Struktura projektu
Copyproject/
│
├── masker2.py         # Przygotowanie masek map pogodowych
├── main_spotter.py    # Główny skrypt do analizy map
├── LH_spotter.py      # Moduł wykrywania systemów L/H
├── x_spotter.py       # Moduł wykrywania znaczników X
├── loader.py          # Skrypt do pobierania map pogodowych
├── connector.py       # Łączenie znaczników X z systemami L/H
│
├── Maps/              # Katalog z mapami źródłowymi
├── masks/             # Wygenerowane maski
├── results/           # Wyniki analizy
└── download/          # Pobrane pliki (ignorowane przez git)
Użycie
Generowanie masek
bashCopypython masker2.py

Przetwarza mapy z folderu Maps
Generuje maski w folderze masks

Analiza map
bashCopypython main_spotter.py

Analizuje maski z folderu masks
Generuje wyniki w folderze results
Tworzy plik weather_systems.csv z danymi o systemach

Pobieranie map
bashCopypython loader.py

Pobiera mapy pogodowe z repozytorium Met Office
Zapisuje w folderze download

Opcje zaawansowane

Tryb debugowy: dodaj flagę --debug do main_spotter.py
Konfiguracja parametrów w plikach .py

Przykładowe wyniki

results/weather_systems.csv: Dane o systemach pogodowych
results/*_result.jpg: Oznaczone obrazy systemów
debug/*_debug.jpg: Szczegółowe obrazy debugowe

Autor
[Twoje imię i nazwisko]
Licencja
MIT
Uwagi

Projekt wymaga dostępu do map pogodowych
Dokładność zależy od jakości i formatu źródłowych map
Folder download jest ignorowany przez system kontroli wersji

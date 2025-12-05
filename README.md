# Vinyl Collector - Vinyl Record Recognition App

Vinyl Collector is a web-based application designed for music collectors to identify vinyl records by simply taking a picture of the album cover. This tool uses computer vision and semantic search to find matches from a database of album covers, providing metadata and helping collectors manage and identify their collections. 

## Features

*   **Image Recognition**: Identify vinyl records by uploading an image or taking a snapshot from your device's camera.
*   **Semantic Search**: Utilizes the DINOv2 model to generate image embeddings and find the most visually similar album covers from the database.
*   **Text Extraction**: Employs PaddleOCR to read text from album art, which can be used for further search and identification.
*   **Top Matches Display**: Shows the top 5 closest matches from the database, including album art, title, artist, year, and style.
*   **Snapshot Gallery**: A gallery view of recently captured or uploaded images, allowing you to review past searches.

## Technology Stack

*   **Backend**: Python, Flask
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Database**: PostgreSQL with the `pgvector` extension for vector similarity search.
*   **Computer Vision & AI**:
    *   `DINOv2` (`facebook/dinov2-small`) for image embedding and semantic search.
    *   `PaddleOCR` for Optical Character Recognition (OCR).
*   **Data Source**: `discogs-client` for fetching album metadata from the Discogs API.

## Project Structure

```
/
├─── camera_app.py             # Main Flask application
├─── camera_functions.py       # Core functions for OCR, embedding, and DB search
├─── extract_images.py         # Script to fetch data from Discogs and populate DB
├─── push_to_db.py             # Script to push metadata and embeddings to PostgreSQL
├─── requirements.txt          # Python dependencies
├─── .env.example              # Example environment variables file
├─── snapshots/                # Directory where uploaded/captured images are saved
└─── covers/                   # Directory where album cover art is downloaded
```

## Setup and Installation

### 1. Prerequisites
*   Python 3.10+
*   PostgreSQL with the `pgvector` extension installed.
*   A Discogs account to get an API token.

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd vinyl-detection
```

### 3. Set up the Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory by copying the example file:
```bash
cp .env.example .env
```
Now, edit the `.env` file with your specific credentials:
```
# PostgreSQL Database URL
# Format: postgresql://<user>:<password>@<host>:<port>/<database>
DB_URL="postgresql://user:password@localhost:5432/vinyl_db"

# Name for the table in the database
TABLE_NAME="album_covers"

# Discogs API Token (generate from your Discogs account settings)
DISCOGS_USER_TOKEN="your_discogs_token_here"
```

### 6. Set up the Database
Make sure your PostgreSQL server is running and you have created the database specified in your `DB_URL`. The `pgvector` extension must also be enabled on the database. You can do this by connecting to your database and running:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Usage

### 1. Populate the Database
Before running the app, you need to populate the database with album data from Discogs. The `extract_images.py` script fetches metadata and cover art, and `push_to_db.py` will create the table and insert the data.

First, run the extraction script. This will fetch data based on styles listed in `cleaned_styles.csv`, download covers, and generate embeddings.
```bash
python .venv/include/vinyl-detection/extract_images.py
```
After this completes, run the push script to populate your PostgreSQL database:
```bash
python .venv/include/vinyl-detection/push_to_db.py
```

### 2. Run the Application
Once the database is populated, you can start the web application:
```bash
python camera_app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000` to start using the app.

## Future Work

*   **Wrap in a Mobile App**: Use a framework like Capacitor or React Native to wrap the web application into a native mobile app for iOS and Android.
*   **User Accounts**: Add user authentication to allow collectors to save their collections.
*   **Refine Search**: Combine semantic search results with OCR text data to improve search accuracy.
*   **Expand Data**: Fetch more comprehensive metadata from Discogs.

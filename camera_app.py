import os
import base64
import datetime
import csv
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from camera_functions import extract_ocr_text, generate_dinov2_embedding, find_top_matches_by_embedding
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'snapshots'
OCR_RESULTS_FILE = 'ocr_results.csv'
EMBEDDINGS_FILE = 'image_embeddings.csv'
MATCHES_FILE = 'matches.csv'
OCR_CONFIDENCE_THRESHOLD = 0.5 # Define confidence threshold here

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Camera App</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; display: flex; flex-direction: column; align-items: center; padding: 20px; margin: 0; }
        h1 { color: #333; }
        .camera-container { position: relative; background: #000; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden; width: 100%; max-width: 640px; }
        video { width: 100%; display: block; }
        .controls { margin-top: 20px; display: flex; gap: 10px; align-items: center; }
        button, .upload-label { padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; transition: background 0.3s; }
        #snapBtn { background-color: #007bff; color: white; }
        #snapBtn:hover { background-color: #0056b3; }
        .upload-label { background-color: #28a745; color: white; }
        .upload-label:hover { background-color: #218838; }
        .status { margin-top: 10px; height: 20px; color: green; font-weight: bold; }
        #results-container { width: 100%; max-width: 640px; margin-top: 20px; }
        .matches-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }
        .match-item { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; padding: 10px; }
        .match-item img { width: 100%; height: auto; border-radius: 4px; }
        .match-item p { margin: 5px 0 0; font-size: 14px; }
        .match-item .title { font-weight: bold; }
        .match-item .artist { color: #555; }
        .match-item .details { font-size: 12px; color: #777; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; width: 100%; max-width: 640px; margin-top: 20px; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .gallery h3 { grid-column: 1 / -1; margin: 0 0 10px 0; font-size: 16px; color: #666; }
        .gallery-item { position: relative; border-radius: 4px; overflow: hidden; border: 1px solid #ddd; cursor: pointer; }
        .gallery-item img { width: 100%; height: 80px; object-fit: cover; display: block; }
        .gallery-item:hover img { opacity: 0.7; }
        .gallery-item-text { position: absolute; bottom: 0; left: 0; width: 100%; background: rgba(0, 0, 0, 0.6); color: white; font-size: 11px; padding: 3px 2px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-sizing: border-box; }
        .modal { display: none; position: fixed; z-index: 100; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.9); }
        .modal-content-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 20px; box-sizing: border-box; }
        .modal-content { display: block; max-width: 80%; max-height: 60vh; border-radius: 4px; }
        #details-content { color: #fff; margin-top: 15px; font-size: 16px; max-width: 80%; text-align: left; white-space: pre-wrap; max-height: 30vh; overflow-y: auto; }
        .close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }
        .close:hover, .close:focus { color: #bbb; text-decoration: none; cursor: pointer; }
        #modal-matches-container { width: 100%; max-width: 80%; margin-top: 15px; }
    </style>
</head>
<body>
    <h1>📷</h1>
    <div class="camera-container"><video id="video" autoplay playsinline></video></div>
    <div class="controls">
        <button id="snapBtn">Take Snapshot</button>
        <label for="uploadInput" class="upload-label">Upload Image</label>
        <input type="file" id="uploadInput" accept="image/*" style="display: none;">
    </div>
    <div id="status" class="status"></div>
    <div id="results-container"></div>
    <div class="gallery" id="gallery"><h3>Recent Snapshots</h3></div>
    <canvas id="canvas" style="display:none;"></canvas>
    <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <div class="modal-content-container">
            <img class="modal-content" id="img01">
            <div id="details-content"></div>
            <div id="modal-matches-container"></div>
        </div>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const snapBtn = document.getElementById('snapBtn');
        const uploadInput = document.getElementById('uploadInput');
        const statusDiv = document.getElementById('status');
        const galleryDiv = document.getElementById('gallery');
        const modal = document.getElementById("imageModal");
        const modalImg = document.getElementById("img01");
        const detailsContent = document.getElementById("details-content");
        const modalMatchesContainer = document.getElementById("modal-matches-container");
        const span = document.getElementsByClassName("close")[0];

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            } catch (err) {
                console.error("Error accessing camera: ", err);
                statusDiv.innerText = "Error: Could not access camera. Check permissions.";
                statusDiv.style.color = "red";
            }
        }

        async function loadGallery() {
            try {
                const response = await fetch('/gallery');
                const items = await response.json();
                const header = galleryDiv.querySelector('h3');
                galleryDiv.innerHTML = '';
                galleryDiv.appendChild(header);
                items.forEach(item => addToGalleryDOM(item.filename, false, item.ocr_results));
            } catch (err) {
                console.error("Error loading gallery:", err);
            }
        }

        function addToGalleryDOM(filename, prepend = false, ocr_results = []) {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'gallery-item';
            itemDiv.onclick = () => openModalWithDetails(filename);

            const img = document.createElement('img');
            img.src = `/snapshots/${filename}`;
            img.alt = filename;
            itemDiv.appendChild(img);

            if (ocr_results && ocr_results.length > 0) {
                const textDiv = document.createElement('div');
                textDiv.className = 'gallery-item-text';
                textDiv.innerText = ocr_results.map(r => r[0]).join(', ');
                itemDiv.appendChild(textDiv);
            }
            
            if (prepend && galleryDiv.children.length > 1) {
                galleryDiv.insertBefore(itemDiv, galleryDiv.children[1]);
            } else {
                galleryDiv.appendChild(itemDiv);
            }
        }
        
        async function openModalWithDetails(filename) {
            modal.style.display = "block";
            modalImg.src = `/snapshots/${filename}`;
            detailsContent.innerText = "Loading details...";
            modalMatchesContainer.innerHTML = ""; // Clear previous matches

            try {
                const response = await fetch(`/details/${filename}`);
                const data = await response.json();
                if (data.error) {
                    detailsContent.innerText = `Error: ${data.error}`;
                } else {
                    const texts = data.ocr_results.map(item => `'${item[0]}' (Confidence: ${item[1].toFixed(2)})`).join('\\n');
                    let ocrText = `Detected Text:\\n`;
                    if (data.ocr_results.length > 0) {
                        ocrText += texts;
                    } else {
                        ocrText += "[] (No text detected or confidence too low)";
                    }
                    const embeddingText = `Embedding: ${data.embedding_summary}`;
                    detailsContent.innerText = `${ocrText}\\n\\n${embeddingText}`;

                    if (data.matches && data.matches.length > 0) {
                        displayMatches(data.matches, 'modal-matches-container');
                    } else {
                        modalMatchesContainer.innerHTML = '<h3>Top Matches</h3><p>No matches found.</p>';
                    }
                }
            } catch (e) {
                detailsContent.innerText = "Could not fetch details.";
            }
        }

        snapBtn.addEventListener('click', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            saveSnapshot(dataURL);
        });

        uploadInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) { return; }

            const reader = new FileReader();
            reader.onload = (e) => {
                const dataURL = e.target.result;
                saveSnapshot(dataURL);
            };
            reader.readAsDataURL(file);
            event.target.value = null;
        });

        function displayMatches(matches, containerId = 'results-container') {
            const resultsContainer = document.getElementById(containerId);
            resultsContainer.innerHTML = '<h3>Top Matches</h3>';
            if (!matches || matches.length === 0) {
                resultsContainer.innerHTML += '<p>No matches found.</p>';
                return;
            }
            const matchesGrid = document.createElement('div');
            matchesGrid.className = 'matches-grid';
            matches.forEach(match => {
                const matchDiv = document.createElement('div');
                matchDiv.className = 'match-item';
                
                const img = document.createElement('img');
                img.src = match.cover_url;
                img.alt = `${match.artist} - ${match.title}`;
                
                const title = document.createElement('p');
                title.className = 'title';
                title.innerText = match.title;
                
                const artist = document.createElement('p');
                artist.className = 'artist';
                artist.innerText = match.artist;

                const details = document.createElement('p');
                details.className = 'details';
                details.innerText = `${match.year || ''} • ${match.style || ''}`;

                matchDiv.appendChild(img);
                matchDiv.appendChild(title);
                matchDiv.appendChild(artist);
                matchDiv.appendChild(details);
                matchesGrid.appendChild(matchDiv);
            });
            resultsContainer.appendChild(matchesGrid);
        }

        async function saveSnapshot(base64Data) {
            statusDiv.innerText = "Saving and processing...";
            document.getElementById('results-container').innerHTML = ''; // Clear previous results
            try {
                const response = await fetch('/save_snapshot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                const result = await response.json();
                if (result.success) {
                    statusDiv.innerText = "Snapshot saved: " + result.filename;
                    statusDiv.style.color = "green";
                    addToGalleryDOM(result.filename, true, result.ocr_results);
                    displayMatches(result.matches);
                    setTimeout(() => { statusDiv.innerText = ""; }, 3000);
                } else {
                    statusDiv.innerText = `Error: ${result.error}`;
                    statusDiv.style.color = "red";
                }
            } catch (error) {
                console.error('Error:', error);
                statusDiv.innerText = "Network error during save.";
            }
        }

        span.onclick = function() { modal.style.display = "none"; }
        window.onclick = function(event) {
            if (event.target == modal) { modal.style.display = "none"; }
        }

        startCamera();
        loadGallery();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image data found'}), 400

    try:
        header, encoded = data['image'].split(",", 1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(encoded))
        
        # --- Run Post-Processing ---
        # 1. OCR
        raw_ocr_result = extract_ocr_text(filepath)
        print(f"Raw OCR result from camera_functions: {raw_ocr_result}") # Keep this for debugging

        processed_ocr_results = []
        if isinstance(raw_ocr_result, list) and raw_ocr_result and isinstance(raw_ocr_result[0], dict):
            ocr_data = raw_ocr_result[0]
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])

            if rec_texts and rec_scores and len(rec_texts) == len(rec_scores):
                all_results = list(zip(rec_texts, rec_scores))
                processed_ocr_results = [
                    (text, score)
                    for text, score in all_results
                    if score > OCR_CONFIDENCE_THRESHOLD
                ]
            else:
                print("OCR: No recognized texts or scores found in model output, or mismatch in lengths.")
        elif 'error' in raw_ocr_result: # Handle error from extract_ocr_text
            print(f"OCR Error: {raw_ocr_result['error']}")
            processed_ocr_results = []
        else:
            print("OCR: Raw OCR result was not in expected dictionary format or was empty.")


        print(f"Processed OCR Results for {filename} (after filtering): {processed_ocr_results}")

        if processed_ocr_results and not (isinstance(processed_ocr_results, dict) and 'error' in processed_ocr_results):
            with open(OCR_RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                for text, score in processed_ocr_results:
                    writer.writerow([filename, text, score])
        
        # 2. Embedding and Matching
        embedding = generate_dinov2_embedding(filepath)
        matches = []
        if embedding is not None:
            with open(EMBEDDINGS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename] + embedding.tolist())
            
            matches = find_top_matches_by_embedding(embedding, top_k=5)
            if matches:
                with open(MATCHES_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for match in matches:
                        writer.writerow([
                            filename,
                            match.get('title'),
                            match.get('artist'),
                            match.get('cover_url'),
                            match.get('year'),
                            match.get('style'),
                            match.get('discogs_url'),
                            match.get('distance')
                        ])

        print(f"Saved and processed: {filepath}")
        return jsonify({
            'success': True,
            'filename': filename,
            'ocr_results': processed_ocr_results or [],
            'matches': matches or []
        })

    except Exception as e:
        print(f"Error saving image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/details/<filename>')
def get_details(filename):
    """Returns OCR results, embedding details, and matches for a given filename."""
    try:
        # OCR Results
        ocr_results = []
        if os.path.exists(OCR_RESULTS_FILE):
            with open(OCR_RESULTS_FILE, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == filename:
                        ocr_results.append((row[1], float(row[2])))

        # Embedding details
        embedding_summary = "Not found"
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == filename:
                        embedding_values = [f"{float(v):.4f}" for v in row[1:6]]
                        embedding_summary = f"Exists (first 5 values: {', '.join(embedding_values)}...)"
                        break
        
        # Matches
        matches = []
        if os.path.exists(MATCHES_FILE):
            with open(MATCHES_FILE, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == filename:
                        matches.append({
                            'title': row[1],
                            'artist': row[2],
                            'cover_url': row[3],
                            'year': row[4],
                            'style': row[5],
                            'discogs_url': row[6],
                            'distance': float(row[7])
                        })

        return jsonify({
            'ocr_results': ocr_results,
            'embedding_summary': embedding_summary,
            'matches': matches
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<path:filename>')
def get_snapshot(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/gallery')
def get_gallery_items():
    try:
        upload_folder_path = Path(UPLOAD_FOLDER)
        files = sorted(
            [f for f in upload_folder_path.glob('*.jpg')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        ocr_data = {}
        if os.path.exists(OCR_RESULTS_FILE):
            with open(OCR_RESULTS_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        filename, text, score = row
                        if filename not in ocr_data:
                            ocr_data[filename] = []
                        ocr_data[filename].append((text, float(score)))
        
        gallery_items = []
        for f in files:
            filename = f.name
            gallery_items.append({
                'filename': filename,
                'ocr_results': ocr_data.get(filename, [])
            })

        return jsonify(gallery_items)
    except Exception as e:
        print(f"Error in get_gallery_items: {e}")
        return jsonify([]), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
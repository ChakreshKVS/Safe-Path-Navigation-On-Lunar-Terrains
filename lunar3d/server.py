from flask import Flask, send_from_directory, Response, request, jsonify
from flask_cors import CORS
import mimetypes
import os
from lunar_crop import LunarTerrainGenerator  # Create this class based on your existing code
import rasterio
import numpy as np
from pyproj import Transformer
import cv2
from datetime import datetime

app = Flask(__name__, static_url_path='')  # Add static_url_path=''
CORS(app)  # Enable CORS for all routes

# Add root route to serve index.html
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/Build/<path:filename>')
def serve_build_files(filename):
    return send_from_directory('Build', filename)

@app.route('/<path:filename>')
def serve_file(filename):
    response = send_from_directory('.', filename)
    # Add cache control headers
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

class LunarTerrainGenerator:
    def __init__(self):
        self.file_path = "WAC_ROI_FARSIDE_DUSK_E300N1350_256P.TIF"  # Updated to relative path
        self.output_dir = "generated_terrain"
        os.makedirs(self.output_dir, exist_ok=True)
        os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

    def generate_image(self, lat, lon, crop_size):
        if not (30.0 <= lat <= 31.0):
            raise ValueError("Latitude must be between 30.0° and 31.0°")
        if not (134.0 <= lon <= 136.0):
            raise ValueError("Longitude must be between 134.0° and 136.0°")
        if not (0.1 <= crop_size <= 1.5):
            raise ValueError("Crop size must be between 0.1° and 1.5°")

        lat_min = lat
        lat_max = lat + crop_size
        lon_min = lon
        lon_max = lon + crop_size

        with rasterio.open(self.file_path) as img_tif:
            transformer = Transformer.from_crs("EPSG:4326", img_tif.crs, always_xy=True)
            
            xmin, ymin = transformer.transform(lon_min, lat_max)
            xmax, ymax = transformer.transform(lon_max, lat_min)

            row_min, col_min = img_tif.index(xmin, ymin)
            row_max, col_max = img_tif.index(xmax, ymax)

            row_min, row_max = max(0, row_min), min(img_tif.height, row_max)
            col_min, col_max = max(0, col_min), min(img_tif.width, col_max)

            cropped_image = img_tif.read(1)[row_min:row_max, col_min:col_max]

            if cropped_image.size == 0:
                raise ValueError("No valid pixels in the selected region")

            cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image)) * 255
            cropped_image = cropped_image.astype(np.uint8)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                f"lunar_terrain_lat{lat:.1f}_lon{lon:.1f}_crop{crop_size:.1f}_{timestamp}.png"
            )

            cv2.imwrite(output_path, cropped_image)
            return output_path

# New route for terrain generation
@app.route('/generate-terrain', methods=['POST'])
def generate_terrain():
    try:
        data = request.json
        lat = float(data.get('lat', 30.5))
        lon = float(data.get('lon', 135.0))
        crop_size = float(data.get('crop_size', 1.0))

        generator = LunarTerrainGenerator()
        image_path = generator.generate_image(lat, lon, crop_size)

        response = send_from_directory(
            os.path.dirname(image_path),
            os.path.basename(image_path),
            mimetype='image/png'
        )
        
        # Add cache control headers
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Added debug=True for development

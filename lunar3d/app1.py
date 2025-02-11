from flask import Flask, request, render_template, send_file, jsonify, url_for
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
from npy_prediction import predict_and_save_height_maps_npy, ResNetUNet
import torch
import numpy as np
import plotly.graph_objects as go
import json
from scipy.ndimage import gaussian_filter
from plotly.utils import PlotlyJSONEncoder
from plotly.offline import plot

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'predicted_npy'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetUNet().to(device)
model.load_state_dict(torch.load("./resnet_mode.pth", map_location=device))
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('lunar_explorer.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Generate prediction
            predict_and_save_height_maps_npy(model, UPLOAD_FOLDER, OUTPUT_FOLDER, device)
            
            # Generate plot
            npy_filename = os.path.splitext(filename)[0] + '.npy'
            npy_path = os.path.join(OUTPUT_FOLDER, npy_filename)
            
            # Load the height map
            input_data = np.load(npy_path, allow_pickle=True).item()
            height_map = input_data['height']
            
            # Get the original image data if available
            image_data = None
            if 'color' in input_data:  # Changed from hasattr to dictionary check
                image_data = input_data['color'].tolist()  # Convert to list
            
            # Create 2D plot with explicit figure closing
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(height_map, cmap="terrain")
            plt.colorbar(label="Height")
            plt.title("Height Map")
            plt.axis("off")
            
            # Save 2D plot to memory
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)  # Explicitly close the figure
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Create 3D surface plot with plotly
            fig = create_lunar_terrain_visualization(height_map, image_data)
            
            # Convert to JSON for sending to frontend using PlotlyJSONEncoder
            plot3d_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
            
            # Convert height map to list for JSON serialization
            height_data = height_map.tolist()
            
            return jsonify({
                'success': True,
                'plot': plot_data,
                'plot3d': plot3d_json,
                'npy_filename': npy_filename,
                'height_data': height_data,
                'image_data': image_data,
                'dimensions': {
                    'width': int(height_map.shape[1]),  # Convert to int
                    'height': int(height_map.shape[0]), # Convert to int
                    'min_height': float(height_map.min()),
                    'max_height': float(height_map.max())
                }
            })
        
        except Exception as e:
            print(f"Error in predict route: {str(e)}")  # Add debug print
            return jsonify({'error': str(e)}), 500
        finally:
            plt.close('all')  # Ensure all figures are closed
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download(filename):
    return send_file(
        os.path.join(OUTPUT_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/view3d/<filename>')
def view_3d(filename):
    try:
        npy_path = os.path.join(OUTPUT_FOLDER, filename)
        input_data = np.load(npy_path, allow_pickle=True).item()
        height_map = input_data['height']
        
        # Get original filename without extension
        base_name = filename.replace('.npy', '')
        
        # Direct file paths
        original_image_url = url_for('static', filename=f'temp/{base_name}_original.png')
        height_map_url = url_for('static', filename=f'temp/{base_name}_height.png')
        
        # Generate height map visualization if it doesn't exist
        height_map_path = os.path.join(app.static_folder, 'temp', f'{base_name}_height.png')
        if not os.path.exists(height_map_path):
            os.makedirs(os.path.dirname(height_map_path), exist_ok=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(height_map, cmap='terrain')
            plt.colorbar(label='Height')
            plt.title('Height Map')
            plt.savefig(height_map_path)
            plt.close()
        
        # Generate Plotly figure with correct colorscale
        fig = go.Figure(data=[
            go.Surface(
                z=height_map,
                colorscale='viridis',  # Using a valid colorscale name
                showscale=True
            )
        ])
        
        fig.update_layout(
            title='Lunar Surface Terrain',
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Height",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        plotly_div = plot(fig, output_type='div', include_plotlyjs=True)
        
        dimensions = {
            'width': height_map.shape[1],
            'height': height_map.shape[0],
            'min_height': float(height_map.min()),
            'max_height': float(height_map.max())
        }
        
        return render_template('lunar_3d.html', 
                             height_data=height_map.tolist(),
                             dimensions=dimensions,
                             original_image_url=original_image_url,
                             height_map_url=height_map_url,
                             plotly_div=plotly_div)
                             
    except Exception as e:
        print(f"Error in view_3d: {str(e)}")  # Debug print
        return str(e), 500

def create_lunar_terrain_visualization(height_map, image_data=None):
    # Remove all smoothing operations
    fig = go.Figure()

    # Generate coordinate grids
    n = height_map.shape[0]
    res = 1.0  # Resolution in meters
    xx, yy = np.meshgrid(
        np.arange(0.0, n * res, res),
        np.arange(0.0, n * res, res)
    )

    # Create realistic lunar colorscale
    lunar_colorscale = [
        [0, 'rgb(30, 30, 30)'],      # Dark shadows
        [0.2, 'rgb(60, 60, 60)'],    # Dark lunar surface
        [0.4, 'rgb(90, 90, 90)'],    # Medium lunar surface
        [0.6, 'rgb(120, 120, 120)'], # Light lunar surface
        [0.8, 'rgb(150, 150, 150)'], # Bright features
        [1, 'rgb(180, 180, 180)']    # Peaks and bright spots
    ]

    surface_props = dict(
        x=xx,
        y=yy,
        z=height_map,
        colorscale=lunar_colorscale,
        showscale=True,
        lighting=dict(
            ambient=0.4,    # Ambient light for shadow detail
            diffuse=0.8,    # Strong diffuse for surface detail
            fresnel=0.2,    # Slight edge enhancement
            specular=0.05,  # Minimal specular reflection
            roughness=0.9   # Very rough surface
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="rgb(120, 120, 120)",
                project_z=True
            )
        )
    )

    fig.add_trace(go.Surface(**surface_props))

    # Update layout for realistic space environment
    fig.update_layout(
        scene=dict(
            xaxis_title="x-axis (m)",
            yaxis_title="y-axis (m)",
            zaxis_title="Height (m)",
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1),
                center=dict(x=0, y=0, z=-0.2)
            ),
            bgcolor='rgb(0, 0, 0)'
        ),
        title="Lunar Surface Terrain",
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',    
        font=dict(color='rgb(200, 200, 200)')
    )

    return fig

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True) 
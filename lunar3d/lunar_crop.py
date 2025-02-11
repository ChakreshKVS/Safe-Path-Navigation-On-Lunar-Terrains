import os
import rasterio
import numpy as np
from pyproj import Transformer
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class LunarTerrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Terrain Cropper")
        self.root.geometry("700x600")
        self.root.configure(bg='#1e1e1e')

        # Updated presets with verified working coordinates
        self.presets = {
            "Center": {"lat": 30.5, "lon": 135.0, "crop": 1.0},
            "North Area": {"lat": 31.0, "lon": 135.5, "crop": 0.5},
            "South Area": {"lat": 30.0, "lon": 134.5, "crop": 0.8},
            "Wide View": {"lat": 30.5, "lon": 135.0, "crop": 1.5},
            "Detailed": {"lat": 30.75, "lon": 135.25, "crop": 0.3}
        }

        # File path
        self.file_path = "E:/obbu/new/lunar3d/lunar3d/WAC_ROI_FARSIDE_DUSK_E300N1350_256P.TIF"
        self.output_dir = "cropped_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Preset selection
        ttk.Label(main_frame, text="Select Preset:").grid(row=0, column=0, pady=5, sticky=tk.W)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(main_frame, textvariable=self.preset_var, values=list(self.presets.keys()))
        preset_combo.grid(row=0, column=1, pady=5, padx=5)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)

        # Input fields
        ttk.Label(main_frame, text="Latitude (30.0 to 31.0):").grid(row=1, column=0, pady=5, sticky=tk.W)
        self.lat_entry = ttk.Entry(main_frame)
        self.lat_entry.grid(row=1, column=1, pady=5, padx=5)
        self.lat_entry.insert(0, "30.5")  # Default value

        ttk.Label(main_frame, text="Longitude (134.0 to 136.0):").grid(row=2, column=0, pady=5, sticky=tk.W)
        self.lon_entry = ttk.Entry(main_frame)
        self.lon_entry.grid(row=2, column=1, pady=5, padx=5)
        self.lon_entry.insert(0, "135.0")  # Default value

        ttk.Label(main_frame, text="Crop Size (degrees):").grid(row=3, column=0, pady=5, sticky=tk.W)
        self.crop_entry = ttk.Entry(main_frame)
        self.crop_entry.grid(row=3, column=1, pady=5, padx=5)
        self.crop_entry.insert(0, "1.0")  # Default value

        # Range indicators
        ttk.Label(main_frame, text="Valid ranges for this TIFF:").grid(row=4, column=0, columnspan=2, pady=(20,5), sticky=tk.W)
        ttk.Label(main_frame, text="Latitude: 30.0° to 31.0°").grid(row=5, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(main_frame, text="Longitude: 134.0° to 136.0°").grid(row=6, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(main_frame, text="Crop Size: 0.1° to 1.5°").grid(row=7, column=0, columnspan=2, sticky=tk.W)

        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="Generate Terrain Image", command=self.generate_image)
        self.generate_btn.grid(row=8, column=0, columnspan=2, pady=20)

        # Status display
        self.status_text = tk.Text(main_frame, height=10, width=60, bg='#2e2e2e', fg='white')
        self.status_text.grid(row=9, column=0, columnspan=2, pady=10)

        # Initial status message
        self.log_status("Ready to generate terrain images.")
        self.log_status("Select a preset or enter custom coordinates.")
        self.log_status("Tip: Start with the 'Center' preset for best results.")

    def apply_preset(self, event=None):
        preset = self.presets[self.preset_var.get()]
        self.lat_entry.delete(0, tk.END)
        self.lat_entry.insert(0, str(preset["lat"]))
        self.lon_entry.delete(0, tk.END)
        self.lon_entry.insert(0, str(preset["lon"]))
        self.crop_entry.delete(0, tk.END)
        self.crop_entry.insert(0, str(preset["crop"]))
        self.log_status(f"Applied preset: {self.preset_var.get()}")

    def log_status(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()

    def generate_image(self):
        try:
            # Get and validate inputs
            lat = float(self.lat_entry.get())
            lon = float(self.lon_entry.get())
            crop_size = float(self.crop_entry.get())

            # Updated validation ranges
            if not (30.0 <= lat <= 31.0):
                raise ValueError("Latitude must be between 30.0° and 31.0° for this TIFF file")
            if not (134.0 <= lon <= 136.0):
                raise ValueError("Longitude must be between 134.0° and 136.0° for this TIFF file")
            if not (0.1 <= crop_size <= 1.5):
                raise ValueError("Crop size must be between 0.1° and 1.5° for best results")

            self.log_status("Starting image generation...")
            self.log_status(f"Coordinates: Lat {lat}°, Lon {lon}°, Crop {crop_size}°")

            # Process image
            os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

            lat_min = lat
            lat_max = lat + crop_size
            lon_min = lon
            lon_max = lon + crop_size

            self.log_status("Opening TIFF file...")
            with rasterio.open(self.file_path) as img_tif:
                transformer = Transformer.from_crs("EPSG:4326", img_tif.crs, always_xy=True)
                
                self.log_status("Transforming coordinates...")
                xmin, ymin = transformer.transform(lon_min, lat_max)
                xmax, ymax = transformer.transform(lon_max, lat_min)

                row_min, col_min = img_tif.index(xmin, ymin)
                row_max, col_max = img_tif.index(xmax, ymax)

                row_min, row_max = max(0, row_min), min(img_tif.height, row_max)
                col_min, col_max = max(0, col_min), min(img_tif.width, col_max)

                self.log_status("Reading image data...")
                cropped_image = img_tif.read(1)[row_min:row_max, col_min:col_max]

                if cropped_image.size == 0:
                    raise ValueError("No valid pixels in the selected region. Try a different area.")

                self.log_status("Processing image...")
                cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image)) * 255
                cropped_image = cropped_image.astype(np.uint8)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self.output_dir, 
                    f"lunar_terrain_lat{lat:.1f}_lon{lon:.1f}_crop{crop_size:.1f}_{timestamp}.png"
                )

                cv2.imwrite(output_path, cropped_image)
                self.log_status(f"Image saved successfully: {output_path}")
                messagebox.showinfo("Success", f"Image generated successfully!\nSaved to: {output_path}")

        except Exception as e:
            self.log_status(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

class LunarTerrainGenerator:
    def __init__(self):
        self.file_path = "E:/obbu/new/lunar3d/lunar3d/WAC_ROI_FARSIDE_DUSK_E300N1350_256P.TIF"
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

if __name__ == "__main__":
    root = tk.Tk()
    app = LunarTerrainApp(root)
    root.mainloop() 
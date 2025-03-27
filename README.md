# Safe-Path-Navigation  

## Overview  
Safe-Path-Navigation is a project designed to analyze lunar terrains and determine safe paths for navigation. The system processes high-resolution lunar images, performs terrain classification, and applies pathfinding algorithms to identify optimal routes.  

---

## Setup Instructions  

### **Step 1: Clone the Repository**  
To get started, clone this repository to your local machine using:  
```bash
git clone https://github.com/your-username/Safe-Path-Navigation.git
cd Safe-Path-Navigation
```

### **Step 2: Download the Required TIFF File**  
The project requires a TIFF file, which can be downloaded from the following link:  

🔗 [WAC_ROI_FARSIDE_DUSK_E300N1350_256P](https://wms.lroc.asu.edu/lroc/view_rdr_product/WAC_ROI_FARSIDE_DUSK_E300N1350_256P)

## Running the Application

### **Step 1: Start the Lunar Processing Modules**
Navigate to the lunar3d directory and execute the following commands:
```bash
python server.py
python app.py
python app1.py
```

## **Step 2: Run the Main Application**
Return to the root folder (Safe-Path-Navigation) and run:
```bash
python app.py
```

## **Step 3: Access the Web Interface**
After executing app1.py, a link will be generated. Open this link in a web browser to interact with the system.

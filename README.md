# PS-SMaRT v1.0  
**Persistent Scatterer–Soil Moisture Analysis for Risk & Triggering**

PS-SMaRT v1.0 is an open-source Python model for the **automatic detection and clustering of unstable slopes** by fusing  
Persistent Scatterer Interferometric SAR (PS-InSAR) time-series deformation with **soil-moisture anomalies and terrain indices**.  
It provides both a scripted backend and a graphical interface for landslide-hazard assessment, developed for publication in  
[*Geoscientific Model Development* (GMD)](https://www.geoscientific-model-development.net/).

---

## ✳️ Key Features
- **LOS → downslope projection** of PS-InSAR velocities using incidence, heading, slope, and aspect.  
- **Automated filtering** by slope and velocity thresholds.  
- **DBSCAN clustering** to delineate coherent unstable zones.  
- **Cluster statistics**: mean, variance, extrema, and polygon geometry.  
- **Soil-moisture anomaly overlap** with χ² and Matthews correlation (φ).  
- **Topographic Wetness Index (TWI)** analysis via Welch’s *t*-test.  
- **Hazard index generation** through robust percentile scaling (P5–P95) and blending into  
  *Low / Moderate / High* classes.  
- **Streamlit interface** for interactive runs, visualization, and output management.

---

## ⚙️ Installation
```bash
# 1. Clone the repository
git clone https://github.com/<yourusername>/ps-smart.git
cd ps-smart

# 2. Create environment and install dependencies
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate ps-smart

## Usage
Command-line
python unstable_slope_hazard_pipeline.py --help

Streamlit Application
streamlit run app.py


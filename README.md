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

## 🧩 Repository Structure
ps-smart/
├── app.py # Streamlit web interface
├── unstable_slope_hazard_pipeline.py # Core processing pipeline
├── docs/
│ └── ps_smart_workflow.png # Workflow schematic
├── examples/ # (optional) example/synthetic datasets
├── requirements.txt # Python dependencies
├── LICENSE # A license
├── CITATION.cff # Citation metadata
└── README.md # This file

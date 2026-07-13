# 🚀 Hyperspectral Anomaly Detection Platform

An interactive **Streamlit-based web application** for **unsupervised anomaly detection in hyperspectral images** using **Principal Component Analysis (PCA)** and **One-Class Support Vector Machine (One-Class SVM)**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 Live Demo

🔗 **Try the application here:**

**https://hyperspectral-anomaly-detection-ny9tphe8kl4kcj3tenmn32.streamlit.app/**

---

# 📖 Project Overview

Hyperspectral imaging captures detailed information across hundreds of spectral bands, enabling the identification of materials and objects that cannot be distinguished using conventional RGB images.

Detecting anomalous regions in hyperspectral images is a challenging task due to the high dimensionality of the data and the lack of labeled datasets. This project presents an **unsupervised anomaly detection framework** that automatically identifies unusual regions without requiring any labeled training samples.

The system combines **Principal Component Analysis (PCA)** for dimensionality reduction and **One-Class Support Vector Machine (One-Class SVM)** for anomaly detection. PCA reduces computational complexity while preserving important spectral information, whereas One-Class SVM learns the distribution of normal pixels and detects spectral outliers as anomalies.

The entire workflow is deployed as an interactive **Streamlit web application**, allowing users to upload hyperspectral datasets, visualize anomaly heatmaps, inspect detected regions, and download analysis reports.

---

# ✨ Features

- 📂 Upload MATLAB (.mat) hyperspectral datasets
- 🌈 Automatic RGB image generation
- 📉 PCA-based dimensionality reduction
- 🤖 One-Class SVM anomaly detection
- 🔥 Interactive anomaly heatmap
- 🎯 Bounding box visualization
- 🧭 Region Explorer
- ⚠️ Severity classification (High, Medium, Low)
- 🧠 Interpretation layer for detected anomalies
- 📊 Region statistics table
- 📥 Download anomaly report (CSV)
- 🖼️ Download processed output image
- ⚙️ Multiple detection modes (Strict, Balanced, Sensitive)
- ⏱️ Processing time display

---

# 🏗️ Project Architecture

```
                  Upload .mat File
                         │
                         ▼
          Read Hyperspectral Image Cube
                         │
                         ▼
              RGB Image Generation
                         │
                         ▼
            Data Standardization
                         │
                         ▼
      Principal Component Analysis (PCA)
                         │
                         ▼
       One-Class Support Vector Machine
                         │
                         ▼
          Generate Anomaly Score Map
                         │
                         ▼
        Heatmap Normalization & Threshold
                         │
                         ▼
      Connected Component Region Detection
                         │
                         ▼
     Bounding Boxes & Region Statistics
                         │
                         ▼
        Interactive Streamlit Dashboard
```


# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application Framework |
| NumPy | Numerical Computation |
| SciPy | Reading MATLAB Files |
| Scikit-Learn | PCA & One-Class SVM |
| Scikit-Image | Region Detection |
| Pandas | Data Analysis |
| Matplotlib | Visualization |

---

# 🧠 Methodology

### 1. Data Loading
The uploaded MATLAB (.mat) file is automatically scanned to identify the first valid 3D hyperspectral cube.

### 2. RGB Visualization
Three spectral bands are selected and normalized to generate an RGB representation of the hyperspectral scene.

### 3. Data Preprocessing
The hyperspectral cube is reshaped into a two-dimensional feature matrix where each pixel represents a spectral feature vector. The data is standardized using **StandardScaler**.

### 4. Dimensionality Reduction
Principal Component Analysis (PCA) reduces hundreds of spectral bands into a smaller feature space while preserving the majority of the variance.

### 5. Anomaly Detection
One-Class Support Vector Machine (One-Class SVM) is trained to model the distribution of normal pixels and detect spectral outliers.

### 6. Heatmap Generation
Anomaly scores are normalized into a heatmap that highlights regions with higher anomaly scores.

### 7. Region Detection
Connected Component Analysis is used to identify anomaly regions, generate bounding boxes, calculate region statistics, and assign severity levels.

### 8. Interactive Dashboard
The Streamlit interface presents:

- RGB Visualization
- Heatmap
- Detection Output
- Region Explorer
- Region Statistics
- Downloadable Reports

---

# 📊 Detection Modes

| Mode | Description |
|------|-------------|
| **Strict** | Detects only highly confident anomalies |
| **Balanced** | Recommended for most datasets |
| **Sensitive** | Detects a larger number of possible anomalies |

---

# 📥 Input

Supported file format:

```
.mat
```

The application automatically detects the hyperspectral cube from the uploaded dataset.

---

# 📤 Outputs

The application generates:

- RGB Image
- Anomaly Heatmap
- Detection Output with Bounding Boxes
- Region Statistics Table
- Region Explorer
- CSV Report
- Processed Output Image

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Hyperspectral-Anomaly-Detection.git
```

Navigate to the project directory

```bash
cd Hyperspectral-Anomaly-Detection
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📈 Applications

- Remote Sensing
- Environmental Monitoring
- Agriculture
- Mineral Exploration
- Defense & Surveillance
- Disaster Management
- Industrial Inspection

---

# 🚀 Future Scope

- Deep Learning-based anomaly detection
- Real-time hyperspectral processing
- GPU acceleration
- Object classification after anomaly detection
- Spectral signature visualization
- UAV and satellite data integration
- Cloud deployment for large-scale datasets

---

# 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!

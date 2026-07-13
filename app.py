import streamlit as st
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
import pandas as pd
import time
import io

st.set_page_config(layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🚀 Hyperspectral Anomaly Detection Platform")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Detection Settings")

mode = st.sidebar.selectbox("Detection Mode", ["Balanced", "Sensitive", "Strict"])

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_heatmap_overlay = st.sidebar.checkbox("Overlay Heatmap", True)
max_regions = st.sidebar.slider("Max Regions", 5, 50, 20)

# Mode settings
if mode == "Strict":
    nu = 0.03
    threshold_percentile = 99.5
elif mode == "Balanced":
    nu = 0.05
    threshold_percentile = 98
else:
    nu = 0.08
    threshold_percentile = 95

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload .mat file", type=["mat"])

if file:
    start_time = time.time()

    mat = sio.loadmat(file)

    cube = None
    cube_key = None

    for key in mat.keys():
        if key.startswith("__"):
            continue

        value = mat[key]

        if isinstance(value, np.ndarray) and value.ndim == 3:
            cube = value
            cube_key = key
            break

    if cube is None:
        st.error("No valid 3D hyperspectral cube found in this .mat file.")
        st.stop()

    H, W, B = cube.shape
    st.success(f"Cube Loaded: {H} x {W} x {B}")
    st.write(f"Detected cube variable: {cube_key}")

    # -----------------------------
    # RGB IMAGE
    # -----------------------------
    rgb = cube[:, :, [min(10, B-1), min(20, B-1), min(30, B-1)]]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    # -----------------------------
    # FLATTEN + SAMPLE (SPEED)
    # -----------------------------
    X = cube.reshape(-1, B)

    sample_size = min(20000, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    # -----------------------------
    # SCALING + PCA
    # -----------------------------
    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)

    pca = PCA(n_components=10)
    X_sample_pca = pca.fit_transform(X_sample_scaled)

    # -----------------------------
    # TRAIN OCSVM
    # -----------------------------
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    model.fit(X_sample_pca)

    # -----------------------------
    # FULL DATA TRANSFORM
    # -----------------------------
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    scores = -model.decision_function(X_pca)
    score_map = scores.reshape(H, W)

    # -----------------------------
    # NORMALIZE HEATMAP (FIXED COLORS)
    # -----------------------------
    norm_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)

    threshold = np.percentile(norm_map, threshold_percentile)
    binary_map = norm_map > threshold

    anomaly_percentage = np.mean(binary_map) * 100

    # -----------------------------
    # REGION DETECTION
    # -----------------------------
    labeled = label(binary_map)
    regions = regionprops(labeled)

    region_data = []

    for i, r in enumerate(regions):
        minr, minc, maxr, maxc = r.bbox
        area = r.area

        region_score = np.mean(norm_map[minr:maxr, minc:maxc])

        if region_score > 0.75:
            severity = "High"
            color = "red"
        elif region_score > 0.5:
            severity = "Medium"
            color = "orange"
        else:
            severity = "Low"
            color = "yellow"

                # -------- INTERPRETATION LAYER --------
        if region_score > 0.75:
            interpretation = "Possible Artificial / Metallic Object"
        elif region_score > 0.5:
            interpretation = "Possible Surface Change / Vegetation Stress"
        else:
            interpretation = "Minor Variation / Noise"

        region_data.append({
            "id": i + 1,
            "area": area,
            "bbox": (minr, minc, maxr, maxc),
            "score": round(region_score, 3),
            "severity": severity,
            "interpretation": interpretation,
            "color": color,
            "width": maxc - minc,
            "height": maxr - minr
        })

    region_data = sorted(region_data, key=lambda x: x["score"], reverse=True)

    df = pd.DataFrame(region_data)

    # -----------------------------
    # DASHBOARD UI
    # -----------------------------

    # TOP ROW
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ RGB Image")
        st.image(rgb, width=400)

    with col2:
        st.subheader("🔥 Heatmap")
        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.imshow(norm_map, cmap='jet')
        ax1.axis("off")
        st.pyplot(fig1)

    # MAIN OUTPUT
    st.subheader("🎯 Detection Output")

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.imshow(rgb)

    if show_heatmap_overlay:
        ax2.imshow(norm_map, cmap='jet', alpha=0.4)

    if show_boxes:
        for r in region_data[:max_regions]:
            y1, x1, y2, x2 = r["bbox"]

            rect = plt.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                edgecolor=r["color"],
                facecolor='none',
                linewidth=2
            )
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f"ID:{r['id']}", color=r["color"], fontsize=8)

    ax2.axis("off")
    st.pyplot(fig2)

    # TABLE + EXPLORER
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📋 Region Table")
        st.dataframe(df, use_container_width=True)

    with col4:
        st.subheader("🔍 Region Explorer")

        if len(region_data) > 0:
            selected = st.selectbox(
                "Select Region",
                [r["id"] for r in region_data[:max_regions]],
                key="region_selector"
            )

            r = next(item for item in region_data if item["id"] == selected)

            y1, x1, y2, x2 = r["bbox"]

            crop = rgb[y1:y2, x1:x2]

            # Show image only if region is large enough
            if r["width"] >= 15 and r["height"] >= 15:
                st.image(crop, width=250)

            st.write(f"Area: {r['area']}")
            st.write(f"Size: {r['width']} x {r['height']}")
            st.write(f"Severity: {r['severity']}")
            st.write(f"Interpretation: {r['interpretation']}")
        else:
            st.info("No anomaly regions detected.")

    # DOWNLOAD + SUMMARY
    st.divider()

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 📥 Downloads")

        csv = df.to_csv(index=False).encode()
        st.download_button("Download Report", csv, "anomaly_report.csv")

        fig2.savefig("output.png", bbox_inches='tight')
        with open("output.png", "rb") as f:
            st.download_button("Download Image", f, "output.png")

    with col6:
        st.markdown("### 📊 Summary")
        st.write(f"Anomaly %: {anomaly_percentage:.2f}%")

        if len(region_data) > 0:
            top = region_data[0]
            st.success(f"Top Region: ID {top['id']} ({top['severity']})")

        st.markdown("""
        ### 🧭 Legend
        - 🔴 High  
        - 🟠 Medium  
        - 🟡 Low  
        """)
        st.markdown("""
        ### 🧠 Interpretation Guide
        - High anomaly → Possible artificial or metallic objects  
        - Medium anomaly → Possible vegetation stress or surface change  
        - Low anomaly → Minor variation or noise  
        """)

    st.success(f"Processing Time: {time.time() - start_time:.2f} sec")
"""
Oil Spill Detection & Vessel Attribution - Live Real-Data Dashboard
========================================================================

Simplified, real-data-only version:
  Tab 1: Spill Detection - upload a real SAR image, run trained U-Net model.
  Tab 2: Live Drift & AIS Attribution - enter a spill location + date,
         fetches REAL Copernicus currents and REAL NOAA AIS data live,
         runs OpenDrift, scores suspect vessels, shows an interactive map.
  Tab 3: Report - downloadable PDF combining both.

No synthetic data, no idealized/simplified fallback modes - everything
in Tab 2 is live-fetched real data (cached locally by date after first
fetch to avoid re-downloading).

Requires live_pipeline.py in the same directory.

Run with:
    streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta

import streamlit as st
import folium
from streamlit_folium import st_folium

import live_pipeline as lp

st.set_page_config(page_title="Oil Spill Attribution System", layout="wide")

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# SECTION 1: SPILL DETECTION (manual upload, real imagery)
# =====================================================================

@st.cache_resource
def load_model(checkpoint_path):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def predict_mask(model, pil_image, threshold=0.5):
    transform = get_transform()
    original = np.array(pil_image.convert("RGB"))
    augmented = transform(image=original)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary_mask = (prob > threshold).astype(np.uint8)
    return original, binary_mask, prob


def extract_geometry(binary_mask):
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {"detected": False}
    largest = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(largest)
    perimeter_px = cv2.arcLength(largest, closed=True)
    x, y, w, h = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (None, None)
    total_area_px = int(binary_mask.sum())
    return {
        "detected": True, "num_regions": len(contours),
        "total_area_pixels": total_area_px, "largest_region_area_pixels": int(area_px),
        "largest_region_perimeter_pixels": round(perimeter_px, 2),
        "bounding_box_xywh": (x, y, w, h),
        "centroid_xy": (round(centroid[0], 2), round(centroid[1], 2)) if centroid[0] else None,
    }


def make_overlay_figure(original, binary_mask, prob):
    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    overlay = original_resized.copy()
    overlay[binary_mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(original_resized, 0.6, overlay, 0.4, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_resized); axes[0].set_title("Original SAR Image"); axes[0].axis("off")
    axes[1].imshow(prob, cmap="viridis"); axes[1].set_title("Predicted Probability"); axes[1].axis("off")
    axes[2].imshow(blended); axes[2].set_title("Detected Oil Spill"); axes[2].axis("off")
    plt.tight_layout()
    return fig


# =====================================================================
# SECTION 2: INTERACTIVE MAP for live drift + AIS results
# =====================================================================

def build_result_map(spill_lat, spill_lon, origin_lat, origin_lon, future_lat, future_lon,
                      b_lons, b_lats, f_lons, f_lats, ais_df, suspect_df, max_particles=40, max_vessels=15):
    m = folium.Map(location=[spill_lat, spill_lon], zoom_start=8, tiles="OpenStreetMap")

    folium.Marker([spill_lat, spill_lon], popup="Reported Spill Location",
                  icon=folium.Icon(color="orange", icon="tint")).add_to(m)
    folium.Marker([origin_lat, origin_lon],
                  popup=f"Estimated Origin ({origin_lat:.4f}, {origin_lon:.4f})",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    folium.Marker([future_lat, future_lon],
                  popup=f"Predicted Future Position ({future_lat:.4f}, {future_lon:.4f})",
                  icon=folium.Icon(color="blue", icon="star")).add_to(m)

    n_particles = b_lons.shape[1]
    sample_idx = np.random.choice(n_particles, size=min(max_particles, n_particles), replace=False)
    for i in sample_idx:
        b_lat_val = b_lats[-1][i]
        b_lon_val = b_lons[-1][i]
        if not np.isnan(b_lat_val) and not np.isnan(b_lon_val):
            folium.CircleMarker([b_lat_val, b_lon_val], radius=2, color="red",
                                 fill=True, fill_opacity=0.5, opacity=0.5).add_to(m)
            
        f_lat_val = f_lats[-1][i]
        f_lon_val = f_lons[-1][i]
        if not np.isnan(f_lat_val) and not np.isnan(f_lon_val):
            folium.CircleMarker([f_lat_val, f_lon_val], radius=2, color="blue",
                                 fill=True, fill_opacity=0.5, opacity=0.5).add_to(m)

    colors = ["green", "purple", "darkred", "cadetblue", "darkorange", "black", "darkgreen"]
    top_mmsi = suspect_df.head(max_vessels)["mmsi"].tolist() if not suspect_df.empty else []
    plotted = ais_df[ais_df["mmsi"].isin(top_mmsi)]
    for i, (mmsi, group) in enumerate(plotted.groupby("mmsi")):
        group = group.sort_values("timestamp")
        score_row = suspect_df[suspect_df["mmsi"] == mmsi]
        score = score_row["suspicion_score"].values[0] if not score_row.empty else None
        vtype = score_row["vessel_type"].values[0] if not score_row.empty else ""
        coords = group[["lat", "lon"]].values.tolist()
        if len(coords) >= 2:
            folium.PolyLine(coords, color=colors[i % len(colors)], weight=3, opacity=0.8,
                             tooltip=f"MMSI {mmsi} ({vtype}) - suspicion score {score}").add_to(m)
        elif len(coords) == 1:
            folium.CircleMarker(coords[0], radius=5, color=colors[i % len(colors)], fill=True,
                                 tooltip=f"MMSI {mmsi} ({vtype}) - suspicion score {score}").add_to(m)

    return m


def make_static_map_figure(spill_lat, spill_lon, origin_lat, origin_lon, future_lat, future_lon,
                            b_lons, b_lats, f_lons, f_lats, ais_df, suspect_df, max_vessels=15):
    """Static matplotlib version used only for the PDF report (folium maps are HTML/JS
    and can't be embedded directly in a PDF)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(b_lons[-1], b_lats[-1], color="red", s=10, alpha=0.4, label="Estimated origin (particles)")
    ax.scatter(f_lons[-1], f_lats[-1], color="blue", s=10, alpha=0.4, label="Predicted spread (particles)")
    ax.scatter([spill_lon], [spill_lat], color="orange", s=200, marker="*",
               edgecolors="black", zorder=5, label="Reported spill location")

    top_mmsi = suspect_df.head(max_vessels)["mmsi"].tolist() if not suspect_df.empty else []
    plotted = ais_df[ais_df["mmsi"].isin(top_mmsi)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(top_mmsi), 1)))
    for (mmsi, group), color in zip(plotted.groupby("mmsi"), colors):
        group = group.sort_values("timestamp")
        ax.plot(group["lon"], group["lat"], marker="o", markersize=2, linewidth=1,
                color=color, alpha=0.8, label=f"MMSI {mmsi}")

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Live Drift & AIS Attribution Result")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# =====================================================================
# SECTION 3: PDF REPORT
# =====================================================================

def add_image_fitted(pdf, img_path, max_width_mm=170):
    from PIL import Image as PILImage
    with PILImage.open(img_path) as im:
        px_w, px_h = im.size
    height_mm = max_width_mm * (px_h / px_w)
    page_bottom = pdf.h - pdf.b_margin
    if pdf.get_y() + height_mm > page_bottom:
        pdf.add_page()
    x = (pdf.w - max_width_mm) / 2
    pdf.image(img_path, x=x, y=pdf.get_y(), w=max_width_mm)
    pdf.set_y(pdf.get_y() + height_mm + 6)


def section_heading(pdf, text):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    from fpdf.enums import XPos, YPos
    pdf.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)


def generate_pdf_report(geometry, drift_info, suspect_df, figs_paths):
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    def mc(pdf, h, text):
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Oil Spill Detection & Attribution Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "Live real-data pipeline: Copernicus Marine currents + NOAA AIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    section_heading(pdf, "1. Spill Detection & Geometry")
    if geometry and geometry.get("detected"):
        mc(pdf, 7,
            f"Regions detected: {geometry['num_regions']}\n"
            f"Total spill area: {geometry['total_area_pixels']} pixels\n"
            f"Largest region area: {geometry['largest_region_area_pixels']} pixels\n"
            f"Largest region perimeter: {geometry['largest_region_perimeter_pixels']} pixels\n"
            f"Bounding box (x,y,w,h): {geometry['bounding_box_xywh']}\n"
            f"Centroid: {geometry['centroid_xy']}\n\n"
            "NOT COMPUTED - Real-world area (m2/km2): requires known sensor ground "
            "resolution, not available for this training dataset.\n"
            "NOT COMPUTED - Slick age estimation: would require multi-temporal image "
            "comparison or spectral oil-thickness analysis; out of scope for this "
            "prototype."
        )
    else:
        mc(pdf, 7, "No detection run / no spill detected for this session.")
    pdf.ln(3)
    if "detection_fig" in figs_paths:
        add_image_fitted(pdf, figs_paths["detection_fig"], max_width_mm=170)

    section_heading(pdf, "2. Live Drift Projection (Real Copernicus Currents)")
    if drift_info:
        mc(pdf, 7,
            f"Reported spill location: ({drift_info['detection_lat']:.4f}, {drift_info['detection_lon']:.4f})\n"
            f"Estimated origin (backtracked): ({drift_info['origin_lat']:.4f}, {drift_info['origin_lon']:.4f})\n"
            f"Predicted future position (+{drift_info['forecast_hours']}h): "
            f"({drift_info['future_lat']:.4f}, {drift_info['future_lon']:.4f})\n\n"
            "NOTE: wind is idealized (constant); real meteorological (ECMWF/CDS) "
            "data was not integrated for this prototype given time constraints. "
            "Ocean currents are real Copernicus Marine data."
        )
    else:
        mc(pdf, 7, "No live drift analysis run for this session.")
    pdf.ln(3)
    if "map_fig" in figs_paths:
        add_image_fitted(pdf, figs_paths["map_fig"], max_width_mm=160)

    section_heading(pdf, "3. Vessel Attribution (Real AIS Data)")
    mc(pdf, 7,
        "AIS source: NOAA MarineCadastre (real, historical AIS records). "
        "NOTE: any specific real-world incident used as a reference scenario may "
        "have an official confirmed cause different from the top-ranked result "
        "shown here (e.g. infrastructure failure rather than vessel discharge). "
        "This demonstrates the system's capability to ingest and score genuine "
        "government AIS records for an analyst-supplied spill location/time."
    )
    pdf.set_font("Helvetica", "", 9)
    if suspect_df is not None and not suspect_df.empty:
        col_widths = [10, 22, 22, 20, 20, 18, 30, 18]
        headers = ["Rank", "MMSI", "Type", "Dist(km)", "Time(h)", "Gap(h)", "TrajDiff", "Score"]
        x_start = pdf.get_x(); y_row = pdf.get_y(); x = x_start
        for h, w in zip(headers, col_widths):
            pdf.set_xy(x, y_row); pdf.cell(w, 8, h, border=1); x += w
        pdf.set_xy(x_start, y_row + 8)
        for _, row in suspect_df.head(10).iterrows():
            values = [row["rank"], row["mmsi"], str(row["vessel_type"])[:10],
                      row["min_distance_km"], row["time_diff_hours"], row["max_ais_gap_hours"],
                      row.get("trajectory_heading_diff_deg", "-"), row["suspicion_score"]]
            x = x_start; y_row = pdf.get_y()
            for val, w in zip(values, col_widths):
                pdf.set_xy(x, y_row); pdf.cell(w, 8, str(val), border=1); x += w
            pdf.set_xy(x_start, y_row + 8)
    else:
        mc(pdf, 7, "No suspect vessels identified within search parameters.")

    return pdf.output(dest="S")


# =====================================================================
# STREAMLIT UI
# =====================================================================

st.title("🛢️ Oil Spill Detection & Vessel Attribution System")
st.caption("Live real-data prototype - real Copernicus ocean currents + real NOAA AIS records")

tab1, tab2, tab3 = st.tabs(["1. Spill Detection", "2. Live Drift & AIS Attribution", "3. Report"])

# ---------------- TAB 1: DETECTION ----------------
with tab1:
    st.header("Oil Spill Detection from SAR Imagery")
    checkpoint_file = st.text_input("Model checkpoint path", value="best_oil_spill_unet.pth")
    uploaded_image = st.file_uploader("Upload a real SAR image", type=["png", "jpg", "jpeg"])

    if uploaded_image and os.path.exists(checkpoint_file):
        model = load_model(checkpoint_file)
        pil_image = Image.open(uploaded_image)
        original, binary_mask, prob = predict_mask(model, pil_image)
        geometry = extract_geometry(binary_mask)

        fig = make_overlay_figure(original, binary_mask, prob)
        st.pyplot(fig)
        fig.savefig("detection_fig.png", dpi=150)

        st.subheader("Geometric Properties")
        st.json(geometry)
        st.caption("Not computed: real-world area (unknown sensor resolution for this "
                   "dataset) and slick age estimation (would need multi-temporal or "
                   "spectral analysis - out of scope for this prototype).")

        st.session_state["geometry"] = geometry
        st.session_state["detection_fig_path"] = "detection_fig.png"
    elif uploaded_image and not os.path.exists(checkpoint_file):
        st.error(f"Checkpoint file not found at: {checkpoint_file}")
    else:
        st.info("Upload a SAR image and provide the model checkpoint path to run detection.")


# ---------------- TAB 2: LIVE DRIFT & AIS ----------------
with tab2:
    st.header("Live Drift Projection & Vessel Attribution")
    st.caption("Enter a spill location and date/time. Ocean currents (Copernicus Marine) and "
               "vessel traffic (NOAA MarineCadastre AIS) are fetched live and cached locally "
               "by date. Region is fixed to the Gulf of Mexico for AIS coverage.")

    col1, col2 = st.columns(2)
    with col1:
        spill_lat = st.number_input("Spill latitude", value=lp.DEFAULT_DETECTION_POINT[0], format="%.4f")
        spill_lon = st.number_input("Spill longitude", value=lp.DEFAULT_DETECTION_POINT[1], format="%.4f")
        spill_date = st.date_input("Spill/detection date", value=datetime(2023, 11, 16))
        spill_time_input = st.time_input("Spill/detection time (UTC)", value=datetime(2023, 11, 16, 12, 0).time())
    with col2:
        backward_hours = st.number_input("Backtrack hours (to estimate origin)", value=14.0, min_value=1.0)
        forward_hours = st.number_input("Forecast forward hours", value=12.0, min_value=1.0)
        st.caption("First run for a new date may take a few minutes (downloading real AIS "
                   "data); repeat runs for the same date reuse the cached files.")

    if st.button("Run Live Analysis"):
        spill_datetime = datetime.combine(spill_date, spill_time_input)
        status = st.empty()

        try:
            results = lp.run_drift_and_ais_pipeline(
                spill_lat, spill_lon, spill_datetime,
                backward_hours=backward_hours, forward_hours=forward_hours,
                progress_callback=lambda msg: status.write(msg),
            )
        except Exception as e:
            st.error(f"Live pipeline failed: {e}")
            st.stop()

        drift_info = results["drift_info"]
        suspect_df = results["suspect_df"]
        ais_df = results["ais_df"]

        st.success("Live analysis complete.")
        st.write(f"**Estimated origin:** ({drift_info['origin_lat']:.4f}, {drift_info['origin_lon']:.4f})")
        st.write(f"**Predicted position in +{forward_hours}h:** "
                 f"({drift_info['future_lat']:.4f}, {drift_info['future_lon']:.4f})")

        st.subheader("Interactive Map")
        result_map = build_result_map(
            spill_lat, spill_lon, drift_info["origin_lat"], drift_info["origin_lon"],
            drift_info["future_lat"], drift_info["future_lon"],
            results["b_lons"], results["b_lats"], results["f_lons"], results["f_lats"],
            ais_df, suspect_df,
        )
        st_folium(result_map, width=1000, height=600, returned_objects=[], key="persistent_drift_map")

        st.subheader("Suspect Vessel Ranking")
        st.dataframe(suspect_df.head(15), width="stretch")
        # Static version for the PDF report (folium maps can't be embedded in PDFs)
        static_fig = make_static_map_figure(
            spill_lat, spill_lon, drift_info["origin_lat"], drift_info["origin_lon"],
            drift_info["future_lat"], drift_info["future_lon"],
            results["b_lons"], results["b_lats"], results["f_lons"], results["f_lats"],
            ais_df, suspect_df,
        )
        static_fig.savefig("map_fig.png", dpi=150)

        st.session_state["drift_info"] = drift_info
        st.session_state["suspect_df"] = suspect_df
        st.session_state["map_fig_path"] = "map_fig.png"


# ---------------- TAB 3: REPORT ----------------
with tab3:
    st.header("Generate PDF Report")
    st.caption("Combines detection (Tab 1) and live drift/AIS results (Tab 2) into a downloadable report.")

    if st.button("Generate Report"):
        geometry = st.session_state.get("geometry")
        drift_info = st.session_state.get("drift_info")
        suspect_df = st.session_state.get("suspect_df", pd.DataFrame())

        figs_paths = {}
        for key, path in [("detection_fig", "detection_fig_path"), ("map_fig", "map_fig_path")]:
            if path in st.session_state:
                figs_paths[key] = st.session_state[path]

        pdf_bytes = generate_pdf_report(geometry, drift_info, suspect_df, figs_paths)

        st.download_button(label="Download PDF Report", data=bytes(pdf_bytes),
                            file_name="oil_spill_report.pdf", mime="application/pdf")
        st.success("Report generated - click above to download.")
    else:
        st.info("Run Tab 1 and/or Tab 2 first, then click 'Generate Report'.")

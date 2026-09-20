"""
Live Real-Data Pipeline - Drift Modeling & Vessel Attribution
======================================================================

Given a spill LOCATION (lat/lon) and DATE/TIME - region otherwise fixed
to the Gulf of Mexico for AIS coverage - this module:
  1. Fetches real ocean currents for that date (Copernicus Marine)
  2. Runs OpenDrift (real currents + idealized wind) to backtrack the
     origin and forecast future spread
  3. Fetches real AIS traffic for that date (NOAA MarineCadastre,
     cached locally after first download) and scores suspect vessels

Spill detection (on a manually uploaded real SAR image) is handled
separately in the dashboard - this module focuses on drift + attribution,
which is where live, real-data orchestration is reliable today. (An
earlier version of this module also live-fetched Sentinel-1 imagery via
Sentinel Hub for automatic detection; that path is kept below for
reference/future use, but is no longer part of the main pipeline after
testing showed real distribution-mismatch and no-data-region issues that
need more time to resolve robustly.)

All functions are designed to be called from the Streamlit dashboard,
but also work standalone for testing.
"""
import os
import gdown

# Auto-download the trained model from Google Drive if not present locally.
# Replace YOUR_FILE_ID with the actual ID from your Google Drive share link:
# e.g. https://drive.google.com/file/d/1AbCdEfGhIjK.../view -> ID is "1AbCdEfGhIjK..."
MODEL_DRIVE_FILE_ID = "1FkxI_Ni_cqln7TGXRMSgbJcLBy4vmmfe"   # <-- replace this before deploying
MODEL_PATH = "best_oil_spill_unet.pth"

if not os.path.exists(MODEL_PATH) and MODEL_DRIVE_FILE_ID != "1FkxI_Ni_cqln7TGXRMSgbJcLBy4vmmfe":
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}",
        MODEL_PATH, quiet=False
    )

import numpy as np
import pandas as pd
import torch
import cv2
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# FIXED REGION: Gulf of Mexico (matches training data lineage + AIS
# coverage). Only the spill location/date are meant to vary.
# -------------------------------------------------------------------
REGION_BBOX = [-90.5, 27.5, -88.0, 29.5]  # [min_lon, min_lat, max_lon, max_lat]
DEFAULT_DETECTION_POINT = (28.95, -89.20)  # (lat, lon) - representative point in region

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = "live_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================================
# 1. SENTINEL HUB - LIVE SAR IMAGE FETCH
# =====================================================================
def fetch_sentinel1_image(date, sh_client_id, sh_client_secret, bbox=REGION_BBOX, size=(512, 512)):
    """
    Fetches a real Sentinel-1 VV image + data-validity mask for the given
    date (searches a +/-2 day window around it to allow for revisit gaps).
    Returns (sar_image_uint8, valid_mask_bool) or (None, None) if no
    scene was found.
    """
    from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

    config = SHConfig()
    config.sh_client_id = sh_client_id
    config.sh_client_secret = sh_client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    # Built-in DataCollection defaults to the OLD services.sentinel-hub.com
    # endpoint regardless of config - must be explicitly redefined.
    cdse_s1_iw = DataCollection.SENTINEL1_IW.define_from("cdse_s1_iw", service_url=config.sh_base_url)

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["VV", "dataMask"],
        output: { bands: 2, sampleType: "UINT8" }
      };
    }
    function evaluatePixel(sample) {
      let db = 10 * Math.log10(sample.VV);
      let val = (db + 25) / 25;
      val = Math.max(0, Math.min(1, val));
      return [val * 255, sample.dataMask * 255];
    }
    """

    start = (date - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=2)).strftime("%Y-%m-%d")

    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=cdse_s1_iw,
            time_interval=(start, end),
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    data = request.get_data()
    if not data or data[0] is None:
        return None, None

    image_full = data[0]
    sar_image = image_full[:, :, 0]
    valid_mask = image_full[:, :, 1] > 0
    return sar_image, valid_mask


# =====================================================================
# 2. DETECTION - runs the trained model, masking out no-data regions
# =====================================================================
def load_model(checkpoint_path):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_detection(model, sar_image_uint8, valid_mask):
    """
    sar_image_uint8: single-channel grayscale array from Sentinel Hub.
    valid_mask: boolean array, True where real data exists.
    Returns (binary_mask, prob_map, geometry_dict) with no-data regions
    excluded from both the mask and the geometry stats.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Convert single-channel to 3-channel RGB-like input (model expects 3 channels)
    rgb_image = np.stack([sar_image_uint8] * 3, axis=-1)

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=rgb_image)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (prob > 0.5).astype(np.uint8)

    # Resize valid_mask to match model output resolution, then apply it
    valid_mask_resized = cv2.resize(valid_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
    binary_mask = binary_mask * valid_mask_resized

    geometry = extract_geometry(binary_mask)
    return binary_mask, prob, geometry, valid_mask_resized


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
        "bounding_box_xywh": (int(x), int(y), int(w), int(h)),
        "centroid_xy": (round(centroid[0], 2), round(centroid[1], 2)) if centroid[0] else None,
    }


# =====================================================================
# 3. COPERNICUS MARINE - LIVE CURRENT FETCH (cached by date)
# =====================================================================
def fetch_copernicus_currents(date, bbox=REGION_BBOX):
    nc_path = os.path.join(CACHE_DIR, f"currents_{date.strftime('%Y%m%d')}.nc")
    if os.path.exists(nc_path):
        return nc_path

    import copernicusmarine

    # On Streamlit Cloud: credentials are injected via st.secrets (set in
    # Streamlit Cloud dashboard -> Settings -> Secrets):
    #   copernicus_username = "your_email"
    #   copernicus_password = "your_password"
    # Locally: copernicusmarine.login() stores credentials in a config file
    # that the library reads automatically - no extra code needed here.
    try:
        import streamlit as st
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = st.secrets["copernicus_username"]
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = st.secrets["copernicus_password"]
    except Exception:
        pass  # Running locally - cached login credentials used automatically

    start = (date - timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%S")
    end = (date + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%S")

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=bbox[0], maximum_longitude=bbox[2],
        minimum_latitude=bbox[1], maximum_latitude=bbox[3],
        start_datetime=start, end_datetime=end,
        minimum_depth=0, maximum_depth=1,
        output_filename=os.path.basename(nc_path),
        output_directory=CACHE_DIR,
    )
    return nc_path


# =====================================================================
# 4. OPENDRIFT - real currents + idealized wind
# =====================================================================
def run_opendrift(detection_lat, detection_lon, detection_time, hours, backward,
                   nc_path, num_particles=100, seed_radius_m=500,
                   wind_x_ms=2.0, wind_y_ms=3.0):
    from opendrift.models.openoil import OpenOil
    from opendrift.readers import reader_netCDF_CF_generic, reader_constant

    o = OpenOil(loglevel=50)
    o.add_reader(reader_netCDF_CF_generic.Reader(nc_path))
    o.add_reader(reader_constant.Reader({'x_wind': wind_x_ms, 'y_wind': wind_y_ms}))

    o.seed_elements(lon=detection_lon, lat=detection_lat, time=detection_time,
                     number=num_particles, radius=seed_radius_m)
    step = -900 if backward else 900
    o.run(time_step=step, duration=timedelta(hours=hours))

    lons = np.array(o.get_property('lon')[0])
    lats = np.array(o.get_property('lat')[0])
    return lons, lats


def summarize_positions(lons, lats):
    lon_vals, lat_vals = lons[-1], lats[-1]
    mean_lon, mean_lat = np.nanmean(lon_vals), np.nanmean(lat_vals)
    R = 6371.0
    lat1, lon1 = np.radians(mean_lat), np.radians(mean_lon)
    lat2, lon2 = np.radians(lat_vals), np.radians(lon_vals)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))
    return {"mean_lat": float(mean_lat), "mean_lon": float(mean_lon),
            "spread_radius_km": float(np.nanmax(dist_km))}


# =====================================================================
# 5. REAL AIS - fetch (cached by date) + filter + score
# =====================================================================
def fetch_ais_day_file(date):
    """Downloads the NOAA MarineCadastre daily AIS file if not already cached."""
    date_str = date.strftime("%Y_%m_%d")
    csv_path = os.path.join(CACHE_DIR, f"AIS_{date_str}.csv")
    zip_path = os.path.join(CACHE_DIR, f"AIS_{date_str}.zip")

    if os.path.exists(csv_path):
        return csv_path

    import urllib.request
    import zipfile

    year = date.strftime("%Y")
    url = f"https://chs.coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{date_str}.zip"
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CACHE_DIR)

    if not os.path.exists(csv_path):
        # extracted filename might differ slightly - find any new CSV
        for f in os.listdir(CACHE_DIR):
            if f.startswith(f"AIS_{date_str}") and f.endswith(".csv"):
                csv_path = os.path.join(CACHE_DIR, f)
                break

    return csv_path


def vessel_type_name(code):
    try:
        code = int(code)
    except (ValueError, TypeError):
        return str(code) if pd.notna(code) else "Unknown"
    if code == 30: return "Fishing"
    if code in (31, 32): return "Towing"
    if code == 35: return "Military"
    if code == 36: return "Sailing"
    if code == 37: return "Pleasure Craft"
    if 40 <= code <= 49: return "High-Speed Craft"
    if code == 50: return "Pilot Vessel"
    if code == 51: return "Search and Rescue"
    if code == 52: return "Tug"
    if code == 53: return "Port Tender"
    if code == 55: return "Law Enforcement"
    if 60 <= code <= 69: return "Passenger"
    if 70 <= code <= 79: return "Cargo"
    if 80 <= code <= 89: return "Tanker"
    if 90 <= code <= 99: return "Other"
    return f"Type {code}"


def load_and_filter_ais(csv_path, bbox=REGION_BBOX):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns={"MMSI": "mmsi", "BaseDateTime": "timestamp", "LAT": "lat",
                             "LON": "lon", "SOG": "sog_knots", "COG": "cog_deg",
                             "VesselType": "vessel_type_code"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["lon"] >= bbox[0]) & (df["lon"] <= bbox[2]) &
            (df["lat"] >= bbox[1]) & (df["lat"] <= bbox[3])]
    df["vessel_type"] = df.get("vessel_type_code", np.nan).apply(vessel_type_name)
    return df[["mmsi", "vessel_type", "timestamp", "lat", "lon", "sog_knots", "cog_deg"]]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def score_vessels(ais_df, origin_lat, origin_lon, origin_time,
                   time_window_hours=24, spatial_radius_km=60):
    """Unified scoring: proximity + time + AIS gap + trajectory."""
    window_start = origin_time - timedelta(hours=time_window_hours)
    window_end = origin_time + timedelta(hours=time_window_hours)
    results = []

    for mmsi, group in ais_df.groupby("mmsi"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        in_window = group[(group["timestamp"] >= window_start) & (group["timestamp"] <= window_end)]
        if in_window.empty:
            continue

        distances = haversine_km(in_window["lat"].values, in_window["lon"].values, origin_lat, origin_lon)
        min_distance_km = distances.min()
        if min_distance_km > spatial_radius_km:
            continue

        closest_idx = distances.argmin()
        closest_time = in_window.iloc[closest_idx]["timestamp"]
        time_diff_hours = abs((closest_time - origin_time).total_seconds() / 3600)
        proximity_score = max(0, 1 - (min_distance_km / spatial_radius_km))
        time_score = max(0, 1 - (time_diff_hours / time_window_hours))

        group["time_diff_min"] = group["timestamp"].diff().dt.total_seconds() / 60
        max_gap_min = group["time_diff_min"].max() if len(group) > 1 else 0
        max_gap_hours = max_gap_min / 60 if not pd.isna(max_gap_min) else 0

        gap_near_origin = False
        if len(group) > 1:
            gap_idx = group["time_diff_min"].idxmax()
            if gap_idx > 0:
                gap_start = group.loc[gap_idx - 1, "timestamp"]
                gap_end = group.loc[gap_idx, "timestamp"]
                if gap_start <= origin_time <= gap_end:
                    gap_near_origin = True
        gap_score = min(1.0, max_gap_hours / 6) if gap_near_origin else min(1.0, max_gap_hours / 6) * 0.3

        closest_point = in_window.iloc[closest_idx]
        trajectory_score = 0.0
        heading_diff = None
        if pd.notna(closest_point.get("cog_deg", np.nan)):
            bearing_to_origin = np.degrees(np.arctan2(
                np.sin(np.radians(origin_lon - closest_point["lon"])) * np.cos(np.radians(origin_lat)),
                np.cos(np.radians(closest_point["lat"])) * np.sin(np.radians(origin_lat)) -
                np.sin(np.radians(closest_point["lat"])) * np.cos(np.radians(origin_lat)) *
                np.cos(np.radians(origin_lon - closest_point["lon"]))
            )) % 360
            heading_diff = min(abs(closest_point["cog_deg"] - bearing_to_origin),
                                360 - abs(closest_point["cog_deg"] - bearing_to_origin))
            trajectory_score = max(0, 1 - (heading_diff / 90))

        weights = {"proximity": 0.25, "time": 0.20, "gap": 0.35, "trajectory": 0.20}
        suspicion_score = (weights["proximity"] * proximity_score + weights["time"] * time_score +
                            weights["gap"] * gap_score + weights["trajectory"] * trajectory_score)

        results.append({
            "mmsi": mmsi, "vessel_type": group["vessel_type"].iloc[0],
            "min_distance_km": round(min_distance_km, 2), "time_diff_hours": round(time_diff_hours, 2),
            "max_ais_gap_hours": round(max_gap_hours, 2), "gap_near_origin": gap_near_origin,
            "trajectory_heading_diff_deg": round(heading_diff, 1) if heading_diff is not None else None,
            "suspicion_score": round(suspicion_score, 3),
        })

    df = pd.DataFrame(results).sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    if not df.empty:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


# =====================================================================
# 6. FULL ORCHESTRATION - given a spill location + date, does everything
# =====================================================================
def run_drift_and_ais_pipeline(detection_lat, detection_lon, detection_time,
                                backward_hours=14, forward_hours=12, progress_callback=None):
    """
    detection_lat, detection_lon: where the spill was observed (manual
    input - e.g. from a separately-run detection step, or a known
    incident location).
    detection_time: datetime the spill was observed.

    Fetches real Copernicus currents + real NOAA AIS for the region/date,
    runs OpenDrift backward/forward, and scores suspect vessels.

    progress_callback: optional function(str) called with status updates,
    e.g. a Streamlit st.write, for UI feedback during the (potentially
    multi-minute) live fetch process.
    """
    def report(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    results = {}

    report("Fetching real ocean current data (Copernicus Marine)...")
    nc_path = fetch_copernicus_currents(detection_time)

    report("Running OpenDrift backward simulation (real currents)...")
    b_lons, b_lats = run_opendrift(detection_lat, detection_lon, detection_time,
                                    backward_hours, True, nc_path)
    origin = summarize_positions(b_lons, b_lats)

    report("Running OpenDrift forward simulation (real currents)...")
    f_lons, f_lats = run_opendrift(detection_lat, detection_lon, detection_time,
                                    forward_hours, False, nc_path)
    future = summarize_positions(f_lons, f_lats)

    origin_time = detection_time - timedelta(hours=backward_hours)
    results["drift_info"] = {
        "origin_lat": origin["mean_lat"], "origin_lon": origin["mean_lon"],
        "detection_lat": detection_lat, "detection_lon": detection_lon,
        "future_lat": future["mean_lat"], "future_lon": future["mean_lon"],
        "forecast_hours": forward_hours, "mode": "Real Copernicus Data",
    }
    results["b_lons"], results["b_lats"] = b_lons, b_lats
    results["f_lons"], results["f_lats"] = f_lons, f_lats
    results["origin_time"] = origin_time

    report("Fetching real AIS data (NOAA MarineCadastre - first run for a "
           "given date may take a few minutes; cached after that)...")
    ais_csv_path = fetch_ais_day_file(detection_time)

    report("Filtering and scoring vessels...")
    ais_df = load_and_filter_ais(ais_csv_path)
    scored = score_vessels(ais_df, origin["mean_lat"], origin["mean_lon"], origin_time)
    results["suspect_df"] = scored
    results["ais_df"] = ais_df

    report("Done.")
    return results

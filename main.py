import os
import zipfile
import streamlit as st
import geopandas as gpd
import gdown
import folium
from streamlit_folium import st_folium

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Shapefile Loader Test", layout="wide")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX"
DATA_DIR = "data"
ZIP_DIR = os.path.join(DATA_DIR, "zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "shp")

# ---------------- UI ----------------
st.title("📍 Shapefile Load Test (Google Drive → Streamlit)")
st.caption("Downloading large shapefile at runtime. GitHub stays clean. Brain stays happy.")

# ---------------- HELPERS ----------------
def download_zip_from_drive():
    os.makedirs(ZIP_DIR, exist_ok=True)

    with st.spinner("⬇️ Downloading ZIP from Google Drive..."):
        gdown.download_folder(
            DRIVE_FOLDER_URL,
            output=ZIP_DIR,
            quiet=False,
            use_cookies=False
        )

def extract_zip():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]
    if not zip_files:
        st.error("❌ No ZIP file found in Drive folder")
        st.stop()

    zip_path = os.path.join(ZIP_DIR, zip_files[0])

    with st.spinner("🗜 Extracting shapefile..."):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EXTRACT_DIR)

    return zip_files[0]

def find_shapefile():
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".shp"):
                return os.path.join(root, f)
    return None

# ---------------- PIPELINE ----------------
if st.button("🚀 Download & Load Shapefile"):

    if not os.path.exists(ZIP_DIR):
        download_zip_from_drive()

    zip_name = extract_zip()
    shp_path = find_shapefile()

    if shp_path is None:
        st.error("❌ Shapefile (.shp) not found after extraction")
        st.stop()

    st.success(f"✅ Loaded ZIP: {zip_name}")
    st.info(f"📂 Shapefile path: {shp_path}")

    # ---------------- LOAD SHAPEFILE ----------------
    with st.spinner("🧠 Reading shapefile with GeoPandas..."):
        gdf = gpd.read_file(shp_path)

    st.success(f"✅ Shapefile loaded | Rows: {len(gdf)}")

    # ---------------- PREVIEW ----------------
    st.subheader("🔍 Attribute Preview")
    st.dataframe(gdf.head())

    # ---------------- MAP ----------------
    st.subheader("🗺 Map Preview")

    center = gdf.geometry.centroid
    m = folium.Map(
        location=[center.y.mean(), center.x.mean()],
        zoom_start=5,
        tiles="cartodbpositron"
    )

    folium.GeoJson(
        gdf.sample(min(500, len(gdf))),  # prevent browser meltdown
        name="Pincodes"
    ).add_to(m)

    st_folium(m, height=500)

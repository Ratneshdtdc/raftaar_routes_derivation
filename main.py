import os
import zipfile
import streamlit as st
import geopandas as gpd
import gdown

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Shapefile Loader", layout="wide")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX"
DATA_DIR = "data"
ZIP_DIR = os.path.join(DATA_DIR, "zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "shp")

# ---------------- UI ----------------
st.title("📦 Large Shapefile Loader (No Visualization)")
st.caption("Google Drive → Streamlit → GeoPandas")

# ---------------- HELPERS ----------------
def download_from_drive():
    os.makedirs(ZIP_DIR, exist_ok=True)

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

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)

    return zip_files[0]

def find_shapefile():
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".shp"):
                return os.path.join(root, f)
    return None

@st.cache_data(show_spinner=False)
def load_gdf(shp_path):
    return gpd.read_file(shp_path)

# ---------------- MAIN ----------------
if st.button("🚀 Download & Load Shapefile"):

    with st.spinner("⬇️ Downloading from Google Drive..."):
        download_from_drive()

    zip_name = extract_zip()
    shp_path = find_shapefile()

    if shp_path is None:
        st.error("❌ Shapefile (.shp) not found")
        st.stop()

    st.success(f"✅ ZIP loaded: {zip_name}")
    #st.info(f"📂 Shapefile path: {shp_path}")

    with st.spinner("🧠 Reading shapefile..."):
        gdf = load_gdf(shp_path)

    # ---------------- VALIDATION ----------------
    #st.subheader("✅ Shapefile Summary")

    # col1, col2, col3 = st.columns(3)
    # col1.metric("Rows", len(gdf))
    # col2.metric("Columns", len(gdf.columns))
    # col3.metric("CRS", gdf.crs.srs if gdf.crs else "None")

    # st.subheader("📑 Columns")
    # st.write(list(gdf.columns))

    # st.subheader("🔍 Sample Rows")
    # st.dataframe(gdf.head(10))

    # ---------------- OPTIONAL EXPORT ----------------
    # st.subheader("⬇️ Export Processed Data")

    # if st.button("Export as Parquet (recommended)"):
    #     out_path = os.path.join(DATA_DIR, "pincode_polygons.parquet")
    #     gdf.to_parquet(out_path)
    #     st.success("✅ Parquet file created")

    #     with open(out_path, "rb") as f:
    #         st.download_button(
    #             "Download Parquet",
    #             f,
    #             file_name="pincode_polygons.parquet"
    #         )

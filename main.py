import os
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import gdown
import folium
from shapely.geometry import Point
from streamlit_folium import st_folium

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🛣️ Raftaar Bikers Routing Tool", layout="wide")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX"
DATA_DIR = "data"
ZIP_DIR = f"{DATA_DIR}/zip"
SHP_DIR = f"{DATA_DIR}/shp"

np.random.seed(42)

# ---------------- HELPERS ----------------
def download_and_extract_shapefile():
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(SHP_DIR, exist_ok=True)

    gdown.download_folder(
        DRIVE_FOLDER_URL,
        output=ZIP_DIR,
        quiet=False,
        use_cookies=False
    )

    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]
    zip_path = os.path.join(ZIP_DIR, zip_files[0])

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(SHP_DIR)

    for root, _, files in os.walk(SHP_DIR):
        for f in files:
            if f.endswith(".shp"):
                return os.path.join(root, f)

    raise FileNotFoundError("Shapefile not found")

@st.cache_data
def load_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf

def generate_points(polygon, n, pincode, attrs, start_id):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    cid = start_id

    while len(points) < n:
        pt = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        if polygon.contains(pt):
            cid += 1
            points.append({
                "customer_id": f"CUST{cid:07d}",
                "pincode": pincode,
                "lat": pt.y,
                "lon": pt.x,
                **attrs
            })
    return points, cid

# ---------------- UI ----------------
st.title("📍 Raftaar Bikers Routing Tool")

st.markdown("""
### 📥 Input File Format
Download the template, fill it, and upload back.

- **Pincode** → Serviceable pincode  
- **OPD** → Orders per day  
- **Office Code** → Dark store / branch  
- **lat / long** → Dark store coordinates  
""")

# -------- TEMPLATE DOWNLOAD --------
template = pd.DataFrame(columns=["Pincode", "OPD", "Office Code", "lat", "long"])
st.download_button(
    "⬇️ Download Template",
    template.to_excel(index=False, engine="openpyxl"),
    file_name="input_template.xlsx"
)

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("📤 Upload Filled Template", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df["Pincode"] = df["Pincode"].astype(str)

    st.success("✅ Input file loaded")

    # -------- SHAPEFILE LOAD --------
    with st.spinner("⬇️ Loading India Pincode Shapefile..."):
        shp_path = download_and_extract_shapefile()
        gdf = load_shapefile(shp_path)

    # detect pincode column
    pincode_col = next(col for col in gdf.columns if "PIN" in col.upper())
    gdf[pincode_col] = gdf[pincode_col].astype(str)

    # filter required pincodes
    gdf = gdf[gdf[pincode_col].isin(df["Pincode"])]

    st.info(f"Filtered {len(gdf)} pincode polygons")

    # -------- MERGE METADATA --------
    df = df.rename(columns={"Pincode": "pincode", "Office Code": "facility_code"})
    gdf = gdf.merge(df, left_on=pincode_col, right_on="pincode", how="left")

    # -------- GENERATE CUSTOMER POINTS --------
    all_points = []
    counter = 0

    for _, row in gdf.iterrows():
        if row["OPD"] > 0:
            pts, counter = generate_points(
                row.geometry,
                int(row.OPD),
                row.pincode,
                {
                    "facility_code": row.facility_code,
                    "OPD": row.OPD
                },
                counter
            )
            all_points.extend(pts)

    df_customers = pd.DataFrame(all_points)
    st.success(f"🎯 Generated {len(df_customers)} customer points")

    # -------- MAP --------
    st.subheader("🗺 Dark Store & Customer Distribution")

    m = folium.Map(
        location=[df["lat"].mean(), df["long"].mean()],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    # customer points
    for _, r in df_customers.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    # dark stores
    for _, r in df.iterrows():
        folium.Marker(
            [r.lat, r.long],
            popup=f"Facility: {r.facility_code}",
            icon=folium.Icon(color="red", icon="building")
        ).add_to(m)

    st_folium(m, height=550)

    # -------- DOWNLOAD OUTPUT --------
    st.download_button(
        "⬇️ Download Customer Points CSV",
        df_customers.to_csv(index=False),
        file_name="customer_points.csv"
    )

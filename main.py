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
from io import BytesIO
import openpyxl
import math


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
template = pd.DataFrame(
    columns=["Pincode", "OPD", "Office Code", "lat", "long"]
)

buffer = BytesIO()
template.to_excel(buffer, index=False, engine="openpyxl")
buffer.seek(0)

st.download_button(
    label="⬇️ Download Template",
    data=buffer,
    file_name="input_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

def route_bikers(
    df_customers,
    store_lat,
    store_lon,
    num_bikers,
    speed_kmph,
    service_time,
    shift_minutes,
    max_distance
):
    bikers = [{
        "id": f"B{i+1}",
        "lat": store_lat,
        "lon": store_lon,
        "distance": 0,
        "time": 0,
        "served": [],
        "path": [(store_lat, store_lon)]
    } for i in range(num_bikers)]

    unserved = []

    df_customers = df_customers.copy()
    df_customers["dist"] = df_customers.apply(
        lambda r: haversine(store_lat, store_lon, r.lat, r.lon),
        axis=1
    )

    for _, c in df_customers.sort_values("dist").iterrows():
        assigned = False

        bikers.sort(key=lambda x: len(x["served"]))

        for b in bikers:
            d = haversine(b["lat"], b["lon"], c.lat, c.lon)
            ret = haversine(c.lat, c.lon, store_lat, store_lon)

            t = (d + ret) / speed_kmph * 60 + service_time
            dist = d + ret

            if (
                b["time"] + t <= shift_minutes and
                b["distance"] + dist <= max_distance
            ):
                b["served"].append(c)
                b["time"] += t
                b["distance"] += dist
                b["lat"], b["lon"] = c.lat, c.lon
                b["path"].append((c.lat, c.lon))
                assigned = True
                break

        if not assigned:
            unserved.append(c)

    return bikers, unserved

st.sidebar.header("⚙️ Routing Parameters")

START_TIME = st.sidebar.time_input("Start Time", value=pd.to_datetime("10:00").time())
END_TIME = st.sidebar.time_input("End Time", value=pd.to_datetime("20:00").time())

HANDOVER_TIME = st.sidebar.number_input("Handover Time (mins)", 5, 30, 10)
SPEED_KMPH = st.sidebar.number_input("Speed (km/h)", 5, 30, 15)
MAX_DISTANCE = st.sidebar.number_input("Max Distance per Biker (km)", 10, 200, 70)
NUM_BIKERS = st.sidebar.number_input("Number of Bikers", 1, 20, 2)

SHIFT_MINUTES = (
    pd.Timestamp.combine(pd.Timestamp.today(), END_TIME) -
    pd.Timestamp.combine(pd.Timestamp.today(), START_TIME)
).seconds / 60

# df_customers MUST exist here
# Columns required: lat, lon

st.success(f"Customers loaded: {len(df_customers)}")

if st.button("🚀 Run Routing"):

    bikers, unserved = route_bikers(
        df_customers,
        store_lat,
        store_lon,
        NUM_BIKERS,
        SPEED_KMPH,
        HANDOVER_TIME,
        SHIFT_MINUTES,
        MAX_DISTANCE
    )

served = sum(len(b["served"]) for b in bikers)

st.metric("Customers Served", served)
st.metric("Service %", f"{served / len(df_customers) * 100:.1f}%")
st.metric("Unserved", len(unserved))

m = folium.Map(location=[store_lat, store_lon], zoom_start=12)

for b in bikers:
    folium.PolyLine(b["path"], weight=4).add_to(m)

st_folium(m, height=600)


logs = []
for b in bikers:
    for i, c in enumerate(b["served"], 1):
        logs.append({
            "biker": b["id"],
            "seq": i,
            "lat": c.lat,
            "lon": c.lon
        })

st.download_button(
    "⬇️ Download Biker Log",
    pd.DataFrame(logs).to_csv(index=False),
    "biker_log.csv"
)




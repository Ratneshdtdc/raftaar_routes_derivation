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
    service_time_min,
    shift_minutes,
    max_distance_km
):
    bikers = []
    for i in range(num_bikers):
        bikers.append({
            "biker_id": f"B{i+1}",
            "lat": store_lat,
            "lon": store_lon,
            "distance": 0,
            "time": 0,
            "served": [],
            "path": [(store_lat, store_lon)]
        })

    df_customers["dist_store"] = df_customers.apply(
        lambda r: haversine(store_lat, store_lon, r.lat, r.lon), axis=1
    )
    customers = df_customers.sort_values("dist_store").to_dict("records")

    unserved = []

    for cust in customers:
        assigned = False
        bikers = sorted(bikers, key=lambda x: len(x["served"]))

        for b in bikers:
            d = haversine(b["lat"], b["lon"], cust["lat"], cust["lon"])
            return_d = haversine(cust["lat"], cust["lon"], store_lat, store_lon)

            travel_time = d / speed_kmph * 60
            return_time = return_d / speed_kmph * 60

            new_time = b["time"] + travel_time + service_time_min + return_time
            new_dist = b["distance"] + d + return_d

            if new_time <= shift_minutes and new_dist <= max_distance_km:
                b["served"].append(cust)
                b["time"] += travel_time + service_time_min
                b["distance"] += d
                b["lat"], b["lon"] = cust["lat"], cust["lon"]
                b["path"].append((cust["lat"], cust["lon"]))
                assigned = True
                break

        if not assigned:
            unserved.append(cust)

    # return to store
    for b in bikers:
        back = haversine(b["lat"], b["lon"], store_lat, store_lon)
        b["distance"] += back
        b["time"] += back / speed_kmph * 60
        b["path"].append((store_lat, store_lon))

    return bikers, unserved

served = sum(len(b["served"]) for b in bikers)
total = len(df_customers)

st.metric("Customers Served", served)
st.metric("Service %", f"{served/total*100:.1f}%")

for b in bikers:
    st.write(
        b["biker_id"],
        "| Orders:", len(b["served"]),
        "| Distance (km):", round(b["distance"], 1),
        "| Time (min):", round(b["time"], 1)
    )

colors = ["red", "blue", "green", "purple", "orange"]

m = folium.Map(location=[store_lat, store_lon], zoom_start=12)

# Pincode polygons
folium.GeoJson(
    gdf_filtered,
    name="Pincodes",
    style_function=lambda x: {
        "fillColor": "#dbeafe",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.4,
    }
).add_to(m)

# Dark store
folium.Marker(
    [store_lat, store_lon],
    icon=folium.Icon(color="black", icon="building"),
    popup="Dark Store"
).add_to(m)

# Routes
for i, b in enumerate(bikers):
    folium.PolyLine(
        b["path"],
        color=colors[i % len(colors)],
        weight=4,
        tooltip=b["biker_id"]
    ).add_to(m)

    for c in b["served"]:
        folium.CircleMarker(
            [c["lat"], c["lon"]],
            radius=3,
            color=colors[i % len(colors)],
            fill=True
        ).add_to(m)

st_folium(m, height=600)

logs = []
for b in bikers:
    for i, c in enumerate(b["served"], 1):
        logs.append({
            "biker_id": b["biker_id"],
            "seq": i,
            "customer_id": c["customer_id"],
            "lat": c["lat"],
            "lon": c["lon"]
        })

df_log = pd.DataFrame(logs)

st.download_button(
    "⬇️ Download Biker Journey Log",
    df_log.to_csv(index=False),
    file_name="biker_routes.csv"
)



# ============================================================
# 🛣️ Raftaar – Biker Routing & Planning Tool
# ============================================================

import os
import zipfile
import math
from io import BytesIO

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import gdown
import folium

from shapely.geometry import Point
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🛣️ Raftaar Bikers Routing Tool",
    layout="wide"
)

np.random.seed(42)

if "routing_done" not in st.session_state:
    st.session_state.routing_done = False

if "bikers" not in st.session_state:
    st.session_state.bikers = None

if "unserved" not in st.session_state:
    st.session_state.unserved = None


# ============================================================
# CONSTANTS & PATHS
# ============================================================
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX"
DATA_DIR = "data"
ZIP_DIR = f"{DATA_DIR}/zip"
SHP_DIR = f"{DATA_DIR}/shp"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_and_extract_shapefile():
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(SHP_DIR, exist_ok=True)

    gdown.download_folder(
        DRIVE_FOLDER_URL,
        output=ZIP_DIR,
        quiet=True,
        use_cookies=False
    )

    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError("No ZIP found in Drive folder")

    zip_path = os.path.join(ZIP_DIR, zip_files[0])

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(SHP_DIR)

    for root, _, files in os.walk(SHP_DIR):
        for f in files:
            if f.endswith(".shp"):
                return os.path.join(root, f)

    raise FileNotFoundError("Shapefile not found after extraction")


@st.cache_data
def load_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf

def minutes_to_time(start_dt, minutes):
    return (start_dt + timedelta(minutes=minutes)).strftime("%H:%M")

def generate_points_in_polygon(polygon, n, pincode, attrs, start_id):
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

def compute_angle(lat, lon, c_lat, c_lon):
    return math.atan2(lat - c_lat, lon - c_lon)


def route_bikers_v2(
    df_customers,
    store_lat,
    store_lon,
    num_bikers,
    speed_kmph,
    service_time_min,
    shift_minutes,
    max_distance_km
):
    df = df_customers.copy()

    # --- radial sweep ---
    df["angle"] = df.apply(
        lambda r: compute_angle(r.lat, r.lon, store_lat, store_lon),
        axis=1
    )
    df = df.sort_values("angle").reset_index(drop=True)

    chunks = np.array_split(df, num_bikers)

    bikers = []
    unserved = []

    for i, chunk in enumerate(chunks):

        biker = {
            "id": f"B{i+1}",
            "path": [(store_lat, store_lon)],
            "journey": [],
            "served": [],
            "distance": 0.0,
            "time": 0.0
        }

        cur_lat, cur_lon = store_lat, store_lon
        cur_time = 0.0   # minutes since shift start
        cur_dist = 0.0   # km

        for _, c in chunk.iterrows():

            leg_dist = haversine(cur_lat, cur_lon, c.lat, c.lon)
            leg_time = leg_dist / speed_kmph * 60

            arrival_time = cur_time + leg_time
            delivery_complete = arrival_time + service_time_min

            # ---- TIME feasibility (forward only) ----
            if delivery_complete > shift_minutes:
                unserved.append(c)
                continue

            # ---- DISTANCE feasibility (must allow return) ----
            ret_dist = haversine(c.lat, c.lon, store_lat, store_lon)
            if cur_dist + leg_dist + ret_dist > max_distance_km:
                unserved.append(c)
                continue

            # ---- log journey ----
            biker["journey"].append({
                "from": "STORE" if not biker["journey"] else biker["journey"][-1]["to"],
                "to": c.customer_id,
                "pincode": c.pincode,

                "leg_travel_km": round(leg_dist, 2),
                "leg_travel_time_min": round(leg_time, 1),

                "arrival_time_min": round(arrival_time, 1),
                "delivery_complete_min": round(delivery_complete, 1),

                "cumulative_time_min": round(delivery_complete, 1),
                "cumulative_distance_km": round(cur_dist + leg_dist, 2),

                "lat": c.lat,
                "lon": c.lon
            })

            biker["served"].append(c)
            biker["path"].append((c.lat, c.lon))

            cur_lat, cur_lon = c.lat, c.lon
            cur_time = delivery_complete
            cur_dist += leg_dist

        # ---- FINAL RETURN (guaranteed feasible) ----
        ret_dist = haversine(cur_lat, cur_lon, store_lat, store_lon)
        ret_time = ret_dist / speed_kmph * 60

        biker["journey"].append({
            "from": biker["journey"][-1]["to"] if biker["journey"] else "STORE",
            "to": "STORE",
            "pincode": None,

            "leg_travel_km": round(ret_dist, 2),
            "leg_travel_time_min": round(ret_time, 1),

            "arrival_time_min": round(cur_time + ret_time, 1),
            "delivery_complete_min": None,

            "cumulative_time_min": round(cur_time + ret_time, 1),
            "cumulative_distance_km": round(cur_dist + ret_dist, 2),

            "lat": store_lat,
            "lon": store_lon
        })

        biker["path"].append((store_lat, store_lon))
        biker["time"] = round(cur_time + ret_time, 1)
        biker["distance"] = round(cur_dist + ret_dist, 2)

        bikers.append(biker)

    return bikers, unserved


# ============================================================
# UI – HEADER & TEMPLATE
# ============================================================

st.title("🛣️ Raftaar – Biker Routing & Planning Tool")

st.markdown("""
### 📥 Input File Format
- **Pincode** → Serviceable pincode  
- **OPD** → Orders per day  
- **Office Code** → Dark store / branch  
- **lat** → Dark store latitude  
- **long** → Dark store longitude  
""")

template = pd.DataFrame(columns=["Pincode", "OPD", "Office Code", "lat", "long"])
buffer = BytesIO()
template.to_excel(buffer, index=False)
buffer.seek(0)

st.download_button(
    "⬇️ Download Input Template",
    buffer,
    file_name="input_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader("📤 Upload Filled Template", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df_input = pd.read_excel(uploaded_file)
df_input["Pincode"] = df_input["Pincode"].astype(str)

st.success("✅ Input file loaded")

# ============================================================
# LOAD SHAPEFILE & FILTER PINCODES
# ============================================================

# with st.spinner("⬇️ Loading India Pincode Shapefile..."):
#     shp_path = download_and_extract_shapefile()
#     gdf = load_shapefile(shp_path)

@st.cache_data(show_spinner="⬇️ Loading India Pincode Shapefile...")
def load_pincode_gdf():
    shp_path = download_and_extract_shapefile()
    return load_shapefile(shp_path)

gdf = load_pincode_gdf()

pincode_col = next(col for col in gdf.columns if "PIN" in col.upper())
gdf[pincode_col] = gdf[pincode_col].astype(str)

gdf = gdf[gdf[pincode_col].isin(df_input["Pincode"])]

st.info(f"Filtered {len(gdf)} pincode polygons")

# ============================================================
# MERGE METADATA
# ============================================================

df_input = df_input.rename(
    columns={"Pincode": "pincode", "Office Code": "facility_code"}
)

gdf = gdf.merge(df_input, left_on=pincode_col, right_on="pincode", how="left")

# ============================================================
# GENERATE CUSTOMER POINTS
# ============================================================

all_points = []
counter = 0

for _, row in gdf.iterrows():
    if row.OPD > 0:
        pts, counter = generate_points_in_polygon(
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

# ============================================================
# SIDEBAR – ROUTING PARAMETERS
# ============================================================

st.sidebar.header("⚙️ Routing Parameters")

START_TIME = st.sidebar.time_input("Start Time", pd.to_datetime("10:00").time())
END_TIME = st.sidebar.time_input("End Time", pd.to_datetime("20:00").time())

HANDOVER_TIME = st.sidebar.number_input("Handover Time (mins)", 5, 30, 10)
SPEED_KMPH = st.sidebar.number_input("Speed (km/h)", 5, 30, 15)
MAX_DISTANCE = st.sidebar.number_input("Max Distance per Biker (km)", 10, 200, 70)
NUM_BIKERS = st.sidebar.number_input("Number of Bikers", 1, 20, 2)

SHIFT_MINUTES = (
    pd.Timestamp.combine(pd.Timestamp.today(), END_TIME) -
    pd.Timestamp.combine(pd.Timestamp.today(), START_TIME)
).seconds / 60

store_lat = df_input["lat"].iloc[0]
store_lon = df_input["long"].iloc[0]

# ============================================================
# ROUTING EXECUTION
# ============================================================

# if not st.button("🚀 Run Routing"):
#     st.stop()
if st.button("🚀 Run Routing"):
    bikers, unserved = route_bikers_v2(
    df_customers,
    store_lat,
    store_lon,
    NUM_BIKERS,
    SPEED_KMPH,
    HANDOVER_TIME,
    SHIFT_MINUTES,
    MAX_DISTANCE)


    st.session_state.bikers = bikers
    st.session_state.unserved = unserved
    st.session_state.routing_done = True


# ============================================================
# METRICS
# ============================================================

if st.session_state.routing_done:

    bikers = st.session_state.bikers
    unserved = st.session_state.unserved

    served = sum(len(b["served"]) for b in bikers)

    st.subheader("📊 Routing Summary")
    st.metric("Customers Served", served)
    st.metric("Service %", f"{served / len(df_customers) * 100:.1f}%")
    st.metric("Unserved Customers", len(unserved))


# served = sum(len(b["served"]) for b in bikers)

# st.subheader("📊 Routing Summary")
# st.metric("Customers Served", served)
# st.metric("Service %", f"{served / len(df_customers) * 100:.1f}%")
# st.metric("Unserved Customers", len(unserved))

# ============================================================
# MAP VISUALIZATION
# ============================================================

# ============================================================
# MAP VISUALIZATION
# ============================================================

st.subheader("🗺 Biker Routes & Coverage")

if st.session_state.routing_done:

    bikers = st.session_state.bikers

    m = folium.Map(
        location=[store_lat, store_lon],
        zoom_start=12,
        tiles="cartodbpositron"
    )

    # Dark store
    folium.Marker(
        [store_lat, store_lon],
        popup="Dark Store",
        icon=folium.Icon(color="red", icon="building")
    ).add_to(m)

    colors = ["red", "blue", "green", "purple", "orange"]

    for i, b in enumerate(bikers):
        folium.PolyLine(
            b["path"],
            weight=4,
            color=colors[i % len(colors)],
            tooltip=b["id"]
        ).add_to(m)

    st_folium(m, height=600)



# ============================================================
# DOWNLOAD BIKER LOG
# ============================================================
if (
    st.session_state.routing_done
    and st.session_state.bikers is not None
    and len(st.session_state.bikers) > 0
):

    shift_start_dt = datetime.combine(
        datetime.today(),
        START_TIME
    )

    logs = []

    for b in st.session_state.bikers:
        for seq, step in enumerate(b["journey"], 1):

            logs.append({
                "biker_id": b["id"],
                "sequence": seq,

                "from": step["from"],
                "to": step["to"],
                "pincode": step["pincode"],

                "leg_travel_km": step["leg_travel_km"],
                "leg_travel_time_min": step["leg_travel_time_min"],

                "arrival_time": (
                    minutes_to_time(shift_start_dt, step["arrival_time_min"])
                    if step["arrival_time_min"] is not None else None
                ),

                "delivery_complete_time": (
                    minutes_to_time(shift_start_dt, step["delivery_complete_min"])
                    if step["delivery_complete_min"] is not None else None
                ),

                "cumulative_time_min": step["cumulative_time_min"],
                "cumulative_distance_km": step["cumulative_distance_km"],

                "lat": step["lat"],
                "lon": step["lon"]
            })

    df_logs = pd.DataFrame(logs)

    st.download_button(
        "⬇️ Download Full Biker Journey Log",
        df_logs.to_csv(index=False),
        "biker_journey_detailed_with_time.csv"
    )

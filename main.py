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
from sklearn.cluster import KMeans
from shapely.geometry import Point
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import requests

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
@st.cache_data(show_spinner=False)
def road_geometry(lat1, lon1, lat2, lon2):
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )

    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()

    coords = data["routes"][0]["geometry"]["coordinates"]
    # OSRM gives lon,lat → convert to lat,lon
    return [(lat, lon) for lon, lat in coords]


def get_distance_time(lat1, lon1, lat2, lon2, speed_kmph):
    if USE_ROAD_DISTANCE:
        return road_distance_time(lat1, lon1, lat2, lon2)
    else:
        d = haversine(lat1, lon1, lat2, lon2) * ROAD_MULTIPLIER
        t = d / speed_kmph * 60
        return d, t

@st.cache_data(show_spinner=False)
def road_distance_time(lat1, lon1, lat2, lon2):
    """
    Returns (distance_km, time_min) using OSRM
    """
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}"
            #f"?overview=false"
            f"?overview=full&geometries=geojson"

        )

        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        route = data["routes"][0]
        dist_km = route["distance"] / 1000
        time_min = route["duration"] / 60

        return dist_km, time_min

    except Exception:
        # fallback
        d = haversine(lat1, lon1, lat2, lon2)
        return d * 1.4, (d * 1.4 / SPEED_KMPH) * 60


def assign_preferred_biker(df_customers, num_bikers):
    coords = df_customers[["lat", "lon"]].values

    kmeans = KMeans(
        n_clusters=num_bikers,
        random_state=42,
        n_init=10
    )

    df_customers = df_customers.copy()
    df_customers["preferred_biker"] = kmeans.fit_predict(coords)

    return df_customers


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

def route_bikers_v3(
    df_customers,
    store_lat,
    store_lon,
    num_bikers,
    speed_kmph,
    service_time_min,
    shift_minutes,
    max_distance_km
):
    # =========================================================
    # Initialize bikers
    # =========================================================
    bikers = []
    for i in range(num_bikers):
        bikers.append({
            "id": f"B{i+1}",
            "lat": store_lat,
            "lon": store_lon,
            "time": 0.0,        # minutes since shift start
            "distance": 0.0,    # km
            "path": [(store_lat, store_lon)],
            "journey": [],
            "served": []
        })

    unserved = df_customers.copy().reset_index(drop=True)

    # =========================================================
    # PHASE 1 — Mandatory seeding (1 job per biker if feasible)
    # =========================================================
    for bi, b in enumerate(bikers):
        if unserved.empty:
            break

        best_ci = None
        best_complete = None

        for ci, c in unserved.iterrows():

            #leg_dist = haversine(store_lat, store_lon, c.lat, c.lon)
            #leg_time = leg_dist / speed_kmph * 60
            leg_dist, leg_time = get_distance_time(
                b["lat"], b["lon"], c.lat, c.lon, speed_kmph
            )
            arrival = leg_time
            complete = arrival + service_time_min

            #ret_dist = haversine(c.lat, c.lon, store_lat, store_lon)
            #ret_time = ret_dist / speed_kmph * 60

            ret_dist, ret_time = get_distance_time(
                b["lat"], b["lon"], store_lat, store_lon, speed_kmph
            )


            if (
                complete + ret_time > shift_minutes or
                leg_dist + ret_dist > max_distance_km
            ):
                continue

            if best_ci is None or complete < best_complete:
                best_ci = ci
                best_complete = complete

        if best_ci is None:
            continue

        c = unserved.loc[best_ci]
        #leg_dist = haversine(store_lat, store_lon, c.lat, c.lon)
        #leg_time = leg_dist / speed_kmph * 60

        leg_dist, leg_time = get_distance_time(store_lat, store_lon, c.lat, c.lon,speed_kmph)

        b["journey"].append({
            "from": "STORE",
            "to": c.customer_id,
            "pincode": c.pincode,
            "leg_travel_km": round(leg_dist, 2),
            "leg_travel_time_min": round(leg_time, 1),
            "arrival_time_min": round(leg_time, 1),
            "delivery_complete_min": round(best_complete, 1),
            "cumulative_time_min": round(best_complete, 1),
            "cumulative_distance_km": round(leg_dist, 2),
            "lat": c.lat,
            "lon": c.lon,
            "path_geometry": road_geometry(from_lat, from_lon,to_lat, to_lon)
        })

        b["lat"], b["lon"] = c.lat, c.lon
        b["time"] = best_complete
        b["distance"] += leg_dist
        b["path"].append((c.lat, c.lon))
        b["served"].append(c)

        unserved = unserved.drop(best_ci).reset_index(drop=True)

    # =========================================================
    # PHASE 2 — Time-first greedy assignment
    # =========================================================
    while not unserved.empty:
        best = None  # (complete_time, leg_dist, biker_idx, cust_idx)

        for bi, b in enumerate(bikers):
            for ci, c in unserved.iterrows():

                # leg_dist = haversine(b["lat"], b["lon"], c.lat, c.lon)
                # leg_time = leg_dist / speed_kmph * 60

                leg_dist, leg_time = get_distance_time(
                    b["lat"], b["lon"], c.lat, c.lon, speed_kmph )


                arrival = b["time"] + leg_time
                complete = arrival + service_time_min

                ret_dist = haversine(c.lat, c.lon, store_lat, store_lon)
                ret_time = ret_dist / speed_kmph * 60

                ret_dist, ret_time = get_distance_time(c.lat, c.lon, store_lat, store_lon )
                

                if (
                    complete + ret_time > shift_minutes or
                    b["distance"] + leg_dist + ret_dist > max_distance_km
                ):
                    continue

                candidate = (complete, leg_dist, bi, ci)

                if best is None or candidate < best:
                    best = candidate

        if best is None:
            break

        complete, leg_dist, bi, ci = best
        b = bikers[bi]
        c = unserved.loc[ci]

        arrival = b["time"] + leg_dist / speed_kmph * 60

        b["journey"].append({
            "from": b["journey"][-1]["to"],
            "to": c.customer_id,
            "pincode": c.pincode,
            "leg_travel_km": round(leg_dist, 2),
            "leg_travel_time_min": round(leg_dist / speed_kmph * 60, 1),
            "arrival_time_min": round(arrival, 1),
            "delivery_complete_min": round(complete, 1),
            "cumulative_time_min": round(complete, 1),
            "cumulative_distance_km": round(b["distance"] + leg_dist, 2),
            "lat": c.lat,
            "lon": c.lon
        })

        b["lat"], b["lon"] = c.lat, c.lon
        b["time"] = complete
        b["distance"] += leg_dist
        b["path"].append((c.lat, c.lon))
        b["served"].append(c)

        unserved = unserved.drop(ci).reset_index(drop=True)

    # =========================================================
    # PHASE 3 — Return all bikers to store
    # =========================================================
    for b in bikers:
        #ret_dist = haversine(b["lat"], b["lon"], store_lat, store_lon)
        #ret_time = ret_dist / speed_kmph * 60

        ret_dist, ret_time = get_distance_time(b["lat"], b["lon"], store_lat, store_lon,speed_kmph)


        b["journey"].append({
            "from": b["journey"][-1]["to"] if b["journey"] else "STORE",
            "to": "STORE",
            "pincode": None,
            "leg_travel_km": round(ret_dist, 2),
            "leg_travel_time_min": round(ret_time, 1),
            "arrival_time_min": round(b["time"] + ret_time, 1),
            "delivery_complete_min": None,
            "cumulative_time_min": round(b["time"] + ret_time, 1),
            "cumulative_distance_km": round(b["distance"] + ret_dist, 2),
            "lat": store_lat,
            "lon": store_lon
        })

        b["distance"] += ret_dist
        b["time"] += ret_time
        b["path"].append((store_lat, store_lon))

    return bikers, unserved

    
    
@st.cache_data
def load_dark_store_master():
    return pd.read_csv("Dark Store Lat Long.csv")

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


ds_master = load_dark_store_master()

# ============================================================
# UI – HEADER & TEMPLATE
# ============================================================

st.title("🛣️ Raftaar – Biker Routing & Planning Tool")

st.subheader("🏬 Select Dark Store")

USE_ROAD_DISTANCE = True
ROAD_MULTIPLIER = 1.4


ds_master["ds_display"] = (
    ds_master["Dark Store Code"].astype(str)
    + " - "
    + ds_master["Dark Store Name"].astype(str)
)

ds_display = st.selectbox(
    "Dark Store (Code - Name)",
    ds_master["ds_display"].unique()
)

ds_row = ds_master[ds_master["ds_display"] == ds_display].iloc[0]

store_lat = ds_row["Lat"]
store_lon = ds_row["Long"]
ds_code = ds_row["Dark Store Code"]

st.info(
    f"📍 Selected: {ds_row['Dark Store Name']} "
    f"({ds_code}) | {store_lat:.4f}, {store_lon:.4f}"
)

buffer = BytesIO()
template_cols = [
    "AWB Number",
    "CONSIGNEE ADDRESS LINE 1",
    "CONSIGNEE ADDRESS LINE 2",
    "CONSIGNEE ADDRESS LINE 3",
    "CONSIGNEE ADDRESS LINE 4",
    "CONSIGNEE CITY",
    "CONSIGNEE PINCODE",
    "CUSTOMER LAT",
    "CUSTOMER LONG"
]
template = pd.DataFrame(columns=template_cols)
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

df_input["CONSIGNEE PINCODE"] = df_input["CONSIGNEE PINCODE"].astype(str)

df_input = df_input.rename(columns={
    "CONSIGNEE PINCODE": "pincode",
    "CUSTOMER LAT": "lat",
    "CUSTOMER LONG": "lon",
    "AWB Number": "customer_id"
})

st.success("✅ Input file loaded")


# ============================================================
# LOAD SHAPEFILE & FILTER PINCODES
# ============================================================

# with st.spinner("⬇️ Loading India Pincode Shapefile..."):
#     shp_path = download_and_extract_shapefile()
#     gdf = load_shapefile(shp_path)

missing_geo = df_input["lat"].isna() | df_input["lon"].isna()
df_missing = df_input[missing_geo]
df_present = df_input[~missing_geo]


@st.cache_data(show_spinner="⬇️ Loading India Pincode Shapefile...")
def load_pincode_gdf():
    shp_path = download_and_extract_shapefile()
    return load_shapefile(shp_path)

gdf = load_pincode_gdf()

pincode_col = next(col for col in gdf.columns if "PIN" in col.upper())
gdf[pincode_col] = gdf[pincode_col].astype(str)

gdf = gdf[gdf[pincode_col].isin(df_input["pincode"])]

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

generated_points = []
counter = 0

for _, row in df_missing.iterrows():

    poly = gdf[gdf[pincode_col] == row.pincode].geometry.values

    if len(poly) == 0:
        continue  # unserviceable pincode

    pts, counter = generate_points_in_polygon(
        poly[0],
        1,
        row.pincode,
        {},
        counter
    )

    pt = pts[0]
    row["lat"] = pt["lat"]
    row["lon"] = pt["lon"]

    generated_points.append(row)

df_generated = pd.DataFrame(generated_points)

df_customers = pd.concat([df_present, df_generated], ignore_index=True)

#df_customers = df_customers[["customer_id", "pincode", "lat", "lon"]]
df_customers = df_input.copy()

st.success(f"🎯 Generated {len(df_customers)} customer points")

# ============================================================
# SIDEBAR – ROUTING PARAMETERS
# ============================================================

st.sidebar.header("⚙️ Routing Parameters")

#START_TIME = st.sidebar.time_input("Start Time", pd.to_datetime("10:00").time())
#END_TIME = st.sidebar.time_input("End Time", pd.to_datetime("20:00").time())

#HANDOVER_TIME = st.sidebar.number_input("Handover Time (mins)", 5, 30, 10)
#SPEED_KMPH = st.sidebar.number_input("Speed (km/h)", 5, 30, 15)
#MAX_DISTANCE = st.sidebar.number_input("Max Distance per Biker (km)", 10, 200, 70)

# ============================================================
# ROUTING CONSTANTS (FROZEN)
# ============================================================

START_TIME = pd.to_datetime("10:00").time()
END_TIME   = pd.to_datetime("20:00").time()

HANDOVER_TIME = 10          # minutes per delivery
SPEED_KMPH        = 15          # biker speed km/hr
MAX_DISTANCE   = 70          # per biker per shift

SHIFT_MINUTES = (
    pd.Timestamp.combine(pd.Timestamp.today(), END_TIME) -
    pd.Timestamp.combine(pd.Timestamp.today(), START_TIME)
).seconds / 60

NUM_BIKERS = st.sidebar.number_input("Number of Bikers", 1, 20, 2)

# ============================================================
# ROUTING EXECUTION
# ============================================================

# if not st.button("🚀 Run Routing"):
#     st.stop()
if st.button("🚀 Run Routing"):
    bikers, unserved = route_bikers_v3(
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

    # for i, b in enumerate(bikers):
    #     folium.PolyLine(
    #         b["path"],
    #         weight=4,
    #         color=colors[i % len(colors)],
    #         tooltip=b["id"]
    #     ).add_to(m)
        
    for i, b in enumerate(bikers):
        # folium.PolyLine(
        #     b["path"],
        #     weight=4,
        #     color=colors[i % len(colors)],
        #     tooltip=b["id"]
        # ).add_to(m)
        for step in b["journey"]:
            if step["to"] == "STORE":
                continue
        
            folium.PolyLine(
                step["path_geometry"],
                weight=4,
                color=colors[i % len(colors)],
                opacity=0.8
            ).add_to(m)
    
        for seq, step in enumerate(b["journey"], 1):
            if step["to"] == "STORE":
                continue
    
            folium.Marker(
                location=[step["lat"], step["lon"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size:10pt;
                        color:white;
                        background:{colors[i % len(colors)]};
                        border-radius:50%;
                        width:24px;
                        height:24px;
                        text-align:center;
                        line-height:24px;
                    ">{seq}</div>
                    """
                ),
                tooltip=f"""
                <b>Biker:</b> {b['id']}<br>
                <b>Seq:</b> {seq}<br>
                <b>AWB:</b> {step['to']}<br>
                <b>Pincode:</b> {step['pincode']}<br>
                <b>Arrival:</b> {step['arrival_time_min']} min<br>
                <b>Leg Dist:</b> {step['leg_travel_km']} km
                """
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
    # ============================================================
    # BIKER SUMMARY TABLE
    # ============================================================
    
    summary_rows = []
    
    for b in st.session_state.bikers:
    
        orders = len([x for x in b["journey"] if x["to"] != "STORE"])
        total_time = round(b["time"], 1)
        total_distance = round(b["distance"], 2)
    
        summary_rows.append({
            "biker_id": b["id"],
            "total_orders_delivered": orders,
            "total_distance_km": total_distance,
            "total_time_min": total_time,
            "finish_time": minutes_to_time(shift_start_dt, total_time),
            "avg_time_per_order_min": round(total_time / orders, 1) if orders > 0 else 0,
            "avg_distance_per_order_km": round(total_distance / orders, 2) if orders > 0 else 0
        })
    
    df_biker_summary = pd.DataFrame(summary_rows)

    # ============================================================
    # ENRICH LOG WITH ADDRESS DETAILS
    # ============================================================
    
    address_cols = [
        "customer_id",
        "CONSIGNEE ADDRESS LINE 1",
        "CONSIGNEE ADDRESS LINE 2",
        "CONSIGNEE ADDRESS LINE 3",
        "CONSIGNEE ADDRESS LINE 4",
        "CONSIGNEE CITY",
        "pincode"
    ]
    
    df_addr = df_input[address_cols].drop_duplicates("customer_id")
    
    df_logs = df_logs.merge(
        df_addr,
        left_on="to",
        right_on="customer_id",
        how="left"
    )
    
    df_logs.drop(columns=["customer_id"], inplace=True)


    # st.download_button(
    #     "⬇️ Download Full Biker Journey Log",
    #     df_logs.to_csv(index=False),
    #     "biker_journey_detailed_with_time.csv"
    # )
    

    output = BytesIO()
    
    #with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
    
        # ---------------------------
        # Sheet 1: Full Journey Log
        # ---------------------------
        df_logs.to_excel(
            writer,
            sheet_name="Full_Journey_Log",
            index=False
        )
    
        # ---------------------------
        # One sheet per biker
        # ---------------------------
        for biker_id in df_logs["biker_id"].unique():
    
            df_biker = df_logs[df_logs["biker_id"] == biker_id]
    
            sheet_name = biker_id[:31]  # Excel sheet name limit
    
            df_biker.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )
    
    output.seek(0)
    
    st.download_button(
        "⬇️ Download Biker-wise Excel Report",
        data=output,
        file_name="biker_routing_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.subheader("🧾 Biker-wise Summary")
    st.dataframe(df_biker_summary, use_container_width=True)



    

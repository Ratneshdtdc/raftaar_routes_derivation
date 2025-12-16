import streamlit
import os
import zipfile
import gdown
import geopandas as gpd

st.title("Temp1")

#URL = "https://drive.google.com/uc?id=FILE_ID"
URL = "https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX?usp=sharing"
#https://drive.google.com/drive/folders/1q1sXQq-3F2khJul-p5dfbcriS_pnrnPX?usp=sharing
ZIP_PATH = "IndiaPIN.zip"
EXTRACT_DIR = "data"

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR)

if not os.path.exists(ZIP_PATH):
    gdown.download(URL, ZIP_PATH, quiet=False)

if not os.path.exists(os.path.join(EXTRACT_DIR, "IndiaPIN.shp")):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

gdf = gpd.read_file(f"{EXTRACT_DIR}/IndiaPIN.shp")


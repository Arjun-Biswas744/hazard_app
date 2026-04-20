from flask import Flask, render_template, request
import os
import re
import numpy as np
import pandas as pd
import time

# 🔴 IMPORT YOUR FUNCTION FILE NAME HERE
from hazard_map import generate_hazard_map   # make sure file name is hazard_map.py
from PIL import Image
import base64
from io import BytesIO

def combine_maps(img1_base64, img2_base64):
    img1 = Image.open(BytesIO(base64.b64decode(img1_base64)))
    img2 = Image.open(BytesIO(base64.b64decode(img2_base64)))

    h = max(img1.height, img2.height)

    new_img = Image.new("RGB", (img1.width + img2.width, h), (255, 255, 255))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    buffer = BytesIO()
    new_img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
HAZARD_FOLDER = os.path.join(BASE_DIR, "Hazard curve")



########for urve############
def compute_hazard(return_period, location):
    base_path = os.path.join(app.root_path, "Hazard curve")
    folder_path = os.path.join(base_path, location, str(return_period))
    print("📂 Using folder:", folder_path)

    if os.path.exists(folder_path):
        print("📄 Files inside:", os.listdir(folder_path))
    else:
        print("❌ Folder NOT found!")

    if not os.path.exists(folder_path):
        return {
            "error": f"Informations for location '{location}' is not found"
        }

    lambda_target = 1 / return_period
    all_files = os.listdir(folder_path)

    hazard_files = []

    for file in os.listdir(folder_path):

        # ❌ Skip UHS file (e.g., "475")
        if file == str(int(return_period)):
            continue

        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        name = os.path.splitext(file)[0]

        try:
            float(name)  # only numeric files like 0.01, 0.05
            hazard_files.append(file)
        except:
            continue

    print("✅ Hazard files found:", hazard_files)

    hazard_files = sorted(hazard_files, key=lambda x: float(os.path.splitext(x)[0]))

    hazard_curves = {}
    periods = []
    uhs_values = []

    for file in hazard_files:

        file_path = os.path.join(folder_path, file)

        print(f"📄 Reading: {file_path}")

        data = np.\
            genfromtxt(
            file_path,
            skip_header=2,
            invalid_raise=False
        )

        # remove bad rows
        data = data[~np.isnan(data).any(axis=1)]

        if data.size == 0:
            print(f"❌ Empty file: {file}")
            continue

        print("Shape:", data.shape)

        intensity = data[:, 0]
        P50 = data[:, 1]

        valid = (intensity > 0) & (P50 > 0)
        intensity = intensity[valid]
        P50 = P50[valid]

        P50[P50 >= 1] = 0.999999

        lambda_annual = -np.log(1 - P50) / 50

        hazard_curves[file] = {
            "intensity": intensity.tolist(),
            "lambda": lambda_annual.tolist()
        }
    # ---------------------------
    # Read UHS directly from file
    # ---------------------------
    uhs_file_path = os.path.join(folder_path, str(int(return_period)))
    # ---------------------------
    # Extract Site Coordinate (lon, lat)
    # ---------------------------
    site_lon = None
    site_lat = None

    if os.path.exists(uhs_file_path):
        with open(uhs_file_path, "r") as f:
            first_line = f.readline()

            match = re.search(r"X=([\d.]+),\s*Y=([\d.]+)", first_line)
            if match:
                site_lon = float(match.group(1))
                site_lat = float(match.group(2))

    print("📍 Site Location:", site_lon, site_lat)
    print("📄 UHS file:", uhs_file_path)

    uhs_periods = []
    uhs_values = []

    if os.path.exists(uhs_file_path):

        data = np.genfromtxt(
            uhs_file_path,
            skip_header=2,
            invalid_raise=False
        )

        # clean
        data = data[~np.isnan(data).any(axis=1)]

        print("📊 UHS shape:", data.shape)

        # assuming:
        # column 0 = period
        # column 1 = Sa

        uhs_periods = data[:, 0]
        uhs_values = data[:, 1]

    else:
        print("❌ UHS file not found!")

    return {
        "lambda": lambda_target,
        "site_lon": site_lon,
        "site_lat": site_lat,
        "hazard_curves": hazard_curves,
        "uhs_periods": uhs_periods.tolist() if len(uhs_periods) > 0 else [],
        "uhs_values": uhs_values.tolist() if len(uhs_values) > 0 else []
    }

########for CMS  ############

def compute_cms(return_period, location, spectral_period):

    base_path = os.path.join(app.root_path, "Hazard curve")
    cms_file = os.path.join(
        base_path,
        location,
        str(return_period),
        "CMS",
        str(spectral_period)
    )

    print("📄 CMS file:", cms_file)

    if not os.path.exists(cms_file):
        return {"error": "CMS file not found"}

    # ---- Read file ----

    # ---- Read properly ----
    df = pd.read_csv(cms_file, skiprows=2)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename (important)
    df.rename(columns={
        'Period (sec)': 'Period',
        'CMS Median': 'CMS_Median',
        'CMS SigmaLN': 'CMS_SigmaLN',
        'UHS': 'UHS'
    }, inplace=True)

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop bad rows
    df = df.dropna()

    # Extract
    period = df['Period'].values
    cms_median = df['CMS_Median'].values
    cms_sigma = df['CMS_SigmaLN'].values
    uhs = df['UHS'].values

    # ---- Compute ± sigma (lognormal) ----
    cms_plus = np.exp(np.log(cms_median) + cms_sigma)
    cms_minus = np.exp(np.log(cms_median) - cms_sigma)

    return {
        "period": period.tolist(),
        "cms_median": cms_median.tolist(),
        "cms_plus": cms_plus.tolist(),
        "cms_minus": cms_minus.tolist(),
        "uhs": uhs.tolist()
    }


# -----------------------------------
# Get available locations (folders)
# -----------------------------------
def get_locations():
    return [
        name for name in os.listdir(HAZARD_FOLDER)
        if os.path.isdir(os.path.join(HAZARD_FOLDER, name))
    ]

# -----------------------------------
# Get available periods from XYZ files
# -----------------------------------
# -----------------------------------
# Get available periods from XYZ files based on location & return_period
# -----------------------------------
def get_periods(location, return_period):
    folder = os.path.join(HAZARD_FOLDER, location, str(return_period))
    periods = []

    if not os.path.exists(folder):
        return []

    for f in os.listdir(folder):

        # ❌ Skip UHS file (e.g., "475", "975", "2475")
        if f in ["475", "975", "2475"]:
            continue

        try:
            val = float(f)
            periods.append(val)
        except:
            pass

    return sorted(list(set(periods)))

# -----------------------------------

# Routes
@app.route("/")
def index():
    return "App is working"





###### Routes for urve ################

from flask import jsonify   # add this import

@app.route("/compute_hazard", methods=["POST"])
def compute_hazard_api():
    data = request.json
    rp = int(float(data["return_period"]))
    location = data["location"]

    return jsonify(compute_hazard(rp, location))   # ✅ FIX

###### Routes for CMS  ################
@app.route("/compute_cms", methods=["POST"])
def compute_cms_api():
    data = request.json

    rp = int(float(data["return_period"]))
    location = data["location"]
    sp = str(data["spectral_period"])

    return jsonify(compute_cms(rp, location, sp))


###### Routes for taking time period automatially############
@app.route("/get_periods/<location>/<return_period>")
def periods_api(location, return_period):
    return {"periods": get_periods(location, return_period)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

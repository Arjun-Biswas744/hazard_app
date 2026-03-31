from flask import Flask, render_template, request, jsonify
import os
import numpy as np

app = Flask(__name__)

folder_path = folder_path = os.path.join(app.root_path, "Hazard curve")

def compute_hazard(return_period):

    lambda_target = 1 / return_period

    all_files = os.listdir(folder_path)

    hazard_files = []
    for file in all_files:
        try:
            float(file)
            hazard_files.append(file)
        except:
            pass

    hazard_files = sorted(hazard_files, key=float)

    hazard_curves = {}
    periods = []
    uhs_values = []

    for file in hazard_files:

        file_path = os.path.join(folder_path, file)
        data = np.loadtxt(file_path, skiprows=2)

        intensity = data[:, 0]
        P50 = data[:, 1]

        valid = (intensity > 0) & (P50 > 0)
        intensity = intensity[valid]
        P50 = P50[valid]

        P50 = np.clip(P50, 1e-8, 0.999999)

        lambda_annual = -np.log(1 - P50) / 50
        print(f"File {file}: lambda min = {np.min(lambda_annual):.6f}, lambda max = {np.max(lambda_annual):.6f}")

        hazard_curves[file] = {
            "intensity": intensity.tolist(),
            "lambda": lambda_annual.tolist()
        }

        if np.min(lambda_annual) <= lambda_target <= np.max(lambda_annual):

            Sa_interp = np.interp(
                np.log(lambda_target),
                np.log(lambda_annual[::-1]),
                intensity[::-1]
            )

            periods.append(float(file))
            uhs_values.append(Sa_interp)

    periods = np.array(periods)
    uhs_values = np.array(uhs_values)

    sort_index = np.argsort(periods)

    periods = periods[sort_index]
    uhs_values = uhs_values[sort_index]

    return {
        "lambda": lambda_target,
        "hazard_curves": hazard_curves,
        "uhs_periods": periods.tolist(),
        "uhs_values": uhs_values.tolist()
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/compute", methods=["POST"])
def compute():
    rp = float(request.json["return_period"])
    return jsonify(compute_hazard(rp))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
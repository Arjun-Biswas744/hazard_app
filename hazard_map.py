import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import re
import time
import base64
from io import BytesIO
from shapely.vectorized import contains

# ✅ CACHES
SHAPE_CACHE = {}
POLY_CACHE = {}
MAP_CACHE = {}

def generate_hazard_map(LOCATION, TARGET_PERIOD, return_period="475", subfolder=None, highlight_shape=None, point=None):

    # ================= USER OPTIONS =================
    SHOW_VALUES = False if subfolder == "BD" else True
    VALUES_STEP = 40

    COLORBAR_INSIDE = True
    COLORMAP = "jet"

    INTERP_METHOD = "linear"   # ✅ SAME as your original
    GRID_RESOLUTION = 500      # ✅ SAME
    FIG_SIZE = (1, 1)          # ✅ SAME
    # ===============================================

    # ✅ CACHE RESULT (no recompute if same input)
    cache_key = (LOCATION, TARGET_PERIOD, return_period, subfolder,point)
    if cache_key in MAP_CACHE:
        return MAP_CACHE[cache_key]

    base_dir = os.path.dirname(__file__)
    main_folder = os.path.join(base_dir, "Hazard curve")
    location_folder = os.path.join(main_folder, LOCATION, str(return_period))

    if subfolder:
        location_folder = os.path.join(location_folder, subfolder)

    # ---------------------------
    # Shapefile (CACHED)
    # ---------------------------
    # ---------------------------
    # Shapefile (CACHED)
    # ---------------------------
    shp_files = glob.glob(os.path.join(location_folder, "*.shp"))

    if not shp_files:
        raise FileNotFoundError(f"No shapefile found in {location_folder}")

    shapefile_path = shp_files[0]

    if shapefile_path in SHAPE_CACHE:
        shape = SHAPE_CACHE[shapefile_path]
    else:
        try:
            shape = gpd.read_file(shapefile_path)
        except Exception as e:
            print("🔥 Shapefile load failed:", e)
            return {"error": "Shapefile loading failed"}

        if shape.crs is None:
            shape = shape.set_crs(epsg=4326)
        else:
            shape = shape.to_crs(epsg=4326)

        SHAPE_CACHE[shapefile_path] = shape

    # ---------------------------
    # Polygon (CACHED)
    # ---------------------------
    if shapefile_path in POLY_CACHE:
        dhaka_poly = POLY_CACHE[shapefile_path]
    else:
        dhaka_poly = shape.geometry.union_all()
        POLY_CACHE[shapefile_path] = dhaka_poly

    # ---------------------------
    # Hazard file selection
    # ---------------------------
    files = os.listdir(location_folder)
    period_to_file = {}

    for f in files:
        match = re.search(r"\(([\d.]+)\)", f)
        if match:
            try:
                period = float(match.group(1))
                period_to_file[period] = f
            except:
                continue

    if not period_to_file:
        raise ValueError("No valid hazard map files found")

    closest_period = min(period_to_file.keys(), key=lambda x: abs(x - TARGET_PERIOD))
    selected_file = period_to_file[closest_period]

    xyz_path = os.path.join(location_folder, selected_file)

    # ---------------------------
    # Read data
    # ---------------------------
    data = pd.read_csv(xyz_path, sep=",", names=["lon", "lat", "hazard"])
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    points = np.column_stack([data.lon, data.lat])
    values = data.hazard.values

    # ---------------------------
    # Grid
    # ---------------------------
    minx, miny, maxx, maxy = shape.total_bounds

    grid_x, grid_y = np.mgrid[
        minx:maxx:complex(GRID_RESOLUTION),
        miny:maxy:complex(GRID_RESOLUTION)
    ]

    # ---------------------------
    # Interpolation (SAME STYLE)
    # ---------------------------
    grid_z = griddata(points, values, (grid_x, grid_y), method=INTERP_METHOD)

    # Fill NaN (same behavior as your old logic)
    grid_z_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

    # ---------------------------
    # Mask (FAST)
    # ---------------------------
    mask = contains(dhaka_poly, grid_x, grid_y)
    grid_z[~mask] = np.nan

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    img = ax.imshow(
        grid_z.T,
        origin="lower",
        extent=(minx, maxx, miny, maxy),
        cmap=COLORMAP,
        interpolation="bilinear"
    )

    shape.boundary.plot(ax=ax, color="black", linewidth=0.2)
    # ---------------------------
    # Plot Site Point
    # ---------------------------
    if point is not None:
        lon, lat = point

        ax.plot(
            lon, lat,
            marker='o',
            markersize=2,
            color='red',
            markeredgecolor='black',
            linewidth=0.2,
            zorder=5
        )

    if highlight_shape is not None:
        if highlight_shape.crs is None:
            highlight_shape = highlight_shape.set_crs(epsg=4326)

        highlight_shape = highlight_shape.to_crs(shape.crs)
        highlight_shape.boundary.plot(ax=ax, color="red", linewidth=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # ---------------------------
    # Colorbar (UNCHANGED)
    # ---------------------------
    if COLORBAR_INSIDE:

        if subfolder == "BD":
            cax = inset_axes(ax, width="6%", height="30%",
                             bbox_to_anchor=(1.1, 0.05, 1, 1),
                             bbox_transform=ax.transAxes,
                             loc="lower left", borderpad=0)
        else:
            cax = inset_axes(ax, width="6%", height="30%",
                             bbox_to_anchor=(1.2, 0.05, 1, 1),
                             bbox_transform=ax.transAxes,
                             loc="lower left", borderpad=0)

        cbar = plt.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=1, length=0.5, width=0.1, pad=0.2)
        cbar.outline.set_linewidth(0.1)

        percent_dict = {"475": "10%", "975": "5%", "2475": "2%"}
        percent_text = percent_dict.get(str(return_period), "22%")

        cax.text(
            0.5, 1.05,
            f"Intensity (cm/s²)\n{percent_text} in 50 years",
            ha="center", va="bottom",
            fontsize=0.5, fontweight="bold",
            transform=cax.transAxes
        )

    # ---------------------------
    # Values
    # ---------------------------
    if SHOW_VALUES:
        for i in range(0, grid_x.shape[0], VALUES_STEP):
            for j in range(0, grid_x.shape[1], VALUES_STEP):
                if not np.isnan(grid_z[i, j]):
                    ax.text(grid_x[i, j], grid_y[i, j],
                            f"{grid_z[i, j]:.2f}",
                            ha="center", va="center", fontsize=0.5)

    # ---------------------------
    # Save (SAME QUALITY)
    # ---------------------------
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close()

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ✅ Cache result
    MAP_CACHE[cache_key] = image_base64

    return image_base64

def generate_combined_map(bd_img, orig_img, title_bd, title_orig):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
    import base64

    def format_title(title):
        if " at " in title:
            parts = title.split(" at ")
            return parts[0] + "\n at " + parts[1]
        return title

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # spacing for titles
    plt.subplots_adjust(top=0.80, wspace=0.25)

    fig.text(0.5, 0.5, "→", fontsize=18, ha="center", va="center")

    # Decode images
    bd = mpimg.imread(BytesIO(base64.b64decode(bd_img)))
    orig = mpimg.imread(BytesIO(base64.b64decode(orig_img)))

    ax[0].imshow(bd)
    ax[0].set_title(format_title(title_bd), fontsize=5,weight="bold")
    ax[0].axis("off")

    ax[1].imshow(orig)
    ax[1].set_title(format_title(title_orig), fontsize=5, weight="bold")
    ax[1].axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.3)  # ✅ IMPORTANT
    plt.close()

    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

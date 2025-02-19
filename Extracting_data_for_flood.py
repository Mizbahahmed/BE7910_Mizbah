# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:11:31 2025

@author: Mizbah
"""
import geopandas as gpd
import rasterio
import numpy as np
from pathlib import Path

# -----------------------------
# 1. Define File Paths
# -----------------------------
data_folder = Path(r"D:\LSU_Spring 2025\BE 7901\Group_Project_1\Data (Team 3)\Data (Team 3)")
output_folder = Path(r"D:\LSU_Spring 2025\BE 7901\Group_Project_1\Outputs")
output_folder.mkdir(parents=True, exist_ok=True)

buildings = gpd.read_file(data_folder / "nsi (MultiPoint)").explode(index_parts=False).reset_index(drop=True)

# -----------------------------
# 2. Function to Extract Raster Values
# -----------------------------
def extract_values_from_band(points_gdf, raster_path, band_number, new_col_name):
    """
    Extract values from a specific raster band at point locations.
    """
    with rasterio.open(raster_path) as src:
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)
        
        coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
        values = [val[0] if val[0] != src.nodatavals[band_number - 1] else np.nan
                  for val in src.sample(coords, indexes=band_number)]
        
        points_gdf[new_col_name] = values

    return points_gdf

# -----------------------------
# 3. Extract Elevation Values
# -----------------------------
elevation_path = data_folder / "elevation (SingleBandRaster)/SA_DEM.tif"
buildings = extract_values_from_band(buildings, elevation_path, band_number=1, new_col_name="elevation")

# -----------------------------
# 4. Extract Flood Depths for Multiple Bands
# -----------------------------
flood_path = data_folder / "flood (MultiBandRaster)/SA_FloodDepths.tif"

flood_bands = {
    1: "SA_FloodDepths.tif_Band_1",
    2: "SA_FloodDepths.tif_Band_2",
    3: "SA_FloodDepths.tif_Band_3",
    4: "SA_FloodDepths.tif_Band_4"
}

for band_num, band_name in flood_bands.items():
    buildings = extract_values_from_band(buildings, flood_path, band_number=band_num, new_col_name=f"flood_{band_name}")

# -----------------------------
# 5. Save Results to Shapefile
# -----------------------------
output_path = output_folder / "buildings_with_elevation_flood.shp"

if output_path.exists():
    output_path.unlink()

buildings.to_file(output_path, driver="ESRI Shapefile", index=False)

# -----------------------------
# 6. Print Summary
# -----------------------------
print("Shapefile saved successfully at:", output_path)
print("First 5 rows of processed data:")
print(buildings.head().to_string(max_colwidth=35))

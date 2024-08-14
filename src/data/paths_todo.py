import os

# you can change these paths to match your own folder structure

def get_default_path_dict() -> dict:
    base_folder = ""
    path_dict = {
        "das": os.path.join(base_folder, "das", "das_full.nc"),
        "pylons": os.path.join(base_folder, "das", "pylons.csv"),
        "fiber_raw": os.path.join(base_folder, "das", "export.geojson"),
        "fiber_segments": os.path.join(base_folder, "das", "fiber_segments.geojson"),
        "fiber_points": os.path.join(base_folder, "das", "fiber_points.geojson"),
        "channels": os.path.join(base_folder, "das", "channels.csv"),
        "channels_osm": os.path.join(base_folder, "das", "channels_osm.csv"),
        "analysis": os.path.join(base_folder, "wind", "202401_medium.nc"),
        "kakern_bru": os.path.join(base_folder, "wind", "kakernbru"),
        "leknes_lufthavn": os.path.join(base_folder, "wind", "leknes_lufthavn"),
        "dataset_a": os.path.join(base_folder, "complete_datasets", "a.csv"),
    }
    return path_dict

def get_out_dir() -> str:
    return "results"
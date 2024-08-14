# %%
# import useful libraries
import sys
import numpy as np
import scipy as sp
import pandas as pd
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from shapely.geometry import Point

# import custom libraries
sys.path.append("src")
from data import paths
from visu import map

path_dict = paths.get_default_path_dict()

# define coordinate reference systems
lonlat_crs = ccrs.PlateCarree()
wind_crs = ccrs.LambertConformal(central_longitude=15, central_latitude=63, standard_parallels=(63, 63))

# %% [markdown]
# ## Load OSM data

# %%
path = path_dict["fiber_raw"]
gdf = gpd.read_file(path)
# uncomment the next line for an interactive map
#gdf.explore()

# %% [markdown]
# ## Save a map of the fiber

# %%
gdfc = gdf.copy()
gdfc.to_crs("EPSG:32633", inplace=True)  # utm 33 n, wgs84

extent = map.default_extents["lofoten_das"]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.epsg(32633)})
map.add_map(
    ax=ax,
    extent=extent,
    tile_provider="GoogleTiles",
    alpha=0.6,
    scale_modifier = 0
)
gdfc[gdfc["power"]=="cable"].plot(ax=ax, label="underground or underwater", color="blue")
#gdfc[gdfc["power"]=="tower"].plot(ax=ax, label="tower", color="orange", marker="o", markersize=3)
gdfc[gdfc["power"]=="line"].plot(ax=ax, label="aerial", color="red")
ax.legend(loc="best")
# uncomment the next line for saving the figure
#fig.savefig("fiber_course.png", dpi=300)
plt.close()

# %%
way_gdf = gdf[gdf["power"].isin(["cable", "line"])].copy()

def get_start_end_points(way_gdf):
    start_points = []
    end_points = []
    for way in way_gdf.geometry.values:
        start_points.append(way.coords[0])
        end_points.append(way.coords[-1])
    return start_points, end_points

start_points, end_points = get_start_end_points(way_gdf)
print(len(set(start_points)), len(set(end_points)), len(set(start_points + end_points)))

way_gdf["start_point"] = start_points
way_gdf["end_point"] = end_points
way_gdf["order"] = None

# now we order the ways to have a continuous path
# start with the first way, then find the way that has its end point as start point, etc. then find the way that has its start point as end point, etc.

def order_next_way(way_gdf: gpd.GeoDataFrame, current_order: int):
    current_way = way_gdf[way_gdf["order"]==current_order]
    next_way = way_gdf[(way_gdf["start_point"]==current_way["end_point"].values[0])]
    if len(next_way) == 1:
        way_gdf.loc[next_way.index, "order"] = current_order + 1
        return True
    else:
        return False

def order_previous_way(way_gdf: gpd.GeoDataFrame, current_order: int):
    current_way = way_gdf[way_gdf["order"]==current_order]
    next_way = way_gdf[(way_gdf["end_point"]==current_way["start_point"].values[0])]
    if len(next_way) == 1:
        way_gdf.loc[next_way.index[0], "order"] = current_order - 1
        return True
    else:
        return False

def order_all_ways(way_gdf: gpd.GeoDataFrame):
    current_order = 0
    while order_next_way(way_gdf, current_order):
        current_order += 1
    current_order = 0
    while order_previous_way(way_gdf, current_order):
        current_order -= 1

way_gdf.loc[0, "order"] = 0
current_order = 0
order_all_ways(way_gdf)

way_gdf["order"] -= way_gdf["order"].values.min()
way_gdf.sort_values("order", inplace=True)
way_gdf["old_index"] = way_gdf.index
way_gdf.reset_index(drop=True, inplace=True)

# uncomment this to see the reordering in images
#way_gdf.plot(cmap="viridis")
#gdf[gdf["power"].isin(["cable", "line"])].plot(cmap="viridis")
#way_gdf[way_gdf["order"]==0].plot()

# limitation of data
print(way_gdf.loc[1, "id"], way_gdf.loc[1, "note"])
# overview
print(way_gdf.shape)

# uncomment to save
#way_gdf.to_file(path_dict["fiber_segments"], driver="GeoJSON")
way_gdf.head()

# %% [markdown]
# ## matching channels and location

# %%
# exploding the ways into nodes
points_with_attrs = []
for idx, line in way_gdf.iterrows():
    if line.geometry.geom_type == 'LineString':
        subindex = 0
        for coord in line.geometry.coords:
            point = Point(coord)
            point_attrs = line.drop('geometry').to_dict()
            point_attrs['geometry'] = point
            point_attrs["subindex"] = subindex
            points_with_attrs.append(point_attrs)
            subindex += 1

points_gdf = gpd.GeoDataFrame(points_with_attrs, crs=way_gdf.crs)
print(f"number of points = {len(points_gdf)}, number of different points = {points_gdf["geometry"].nunique()}")

lonlat_points_np = np.array(points_gdf.geometry.get_coordinates())
geodesic = Geodesic()
dist_to_next = geodesic.inverse(lonlat_points_np[:-1], lonlat_points_np[1:])[:, 0]
dist_to_next = np.concatenate([dist_to_next, [0.]], axis=0)
plt.plot(dist_to_next)
plt.xlabel("point")
plt.ylabel("distance to the next point (m)")
plt.show()

# %%
points_gdf.columns

# %%
points_lonlat = np.array(points_gdf.geometry.get_coordinates())
fiber_points_df = pd.DataFrame(dict(
    lon=points_lonlat[:, 0],
    lat=points_lonlat[:, 1],
    way_id=points_gdf.id.values,
    subindex=points_gdf.subindex.values,
    power=points_gdf.power.values,
    location=points_gdf.location.values,
))

fiber_points_df.loc[fiber_points_df["location"].isin([None]), "location"] = "overhead"

# uncomment to save
#fiber_points_df.to_csv(path_dict["fiber_points"], index=False)
fiber_points_df.head()

# %%
points_linloc = np.cumsum(np.concatenate([[0.], dist_to_next], axis=0))[:-1]

n_channels = 1394
channels_spacing = 8.16 * 3
channels = np.arange(0, n_channels)
channels_linloc = channels * channels_spacing + 0.5 * channels_spacing

#find landmark points
boundaries_gdf = points_gdf[points_gdf.duplicated(subset=["geometry"], keep=False)]
line_boundaries_gdf = boundaries_gdf[boundaries_gdf["power"]=="line"]
cable_boundaries_gdf = boundaries_gdf[boundaries_gdf["power"]=="cable"]
boundaries_gdf = boundaries_gdf[
    boundaries_gdf["geometry"].isin(line_boundaries_gdf["geometry"]) 
    & boundaries_gdf["geometry"].isin(cable_boundaries_gdf["geometry"])]
#boundaries_gdf.drop_duplicates(subset=["geometry"], keep="first", inplace=True)
boundaries_gdf = boundaries_gdf[boundaries_gdf["power"]=="line"]

landmark_points = list(boundaries_gdf.index)[:-1]
landmark_channels = [106.5, 348, 464.5, 693.5, 793]

# linear regression
landmark_channels_linloc = np.array(landmark_channels) * channels_spacing + 0.5 * channels_spacing
landmark_points_linloc = points_linloc[landmark_points]
res = sp.stats.linregress(landmark_channels_linloc, landmark_points_linloc)

# estimate the error with leave one out
errors = []
for i in range(len(landmark_points)):
    mask = np.ones(len(landmark_points), dtype=bool)
    mask[i] = False
    error_res = sp.stats.linregress(landmark_channels_linloc[mask], landmark_points_linloc[mask])
    errors.append(landmark_points_linloc[i] - (error_res.intercept + error_res.slope * landmark_channels_linloc[i]))
errors = np.array(errors)
print("Standard error assuming the regression is unbiaised: ", np.sqrt(np.mean(errors**2) * len(errors) / (len(errors) - 1)))
print("Leave one out errors: ", errors)

# plot linear regression results
fig, ax = plt.subplots()
ax.plot(landmark_channels_linloc, landmark_points_linloc, 'o', label='original data')
ax.plot(channels_linloc, res.intercept + res.slope * channels_linloc, 'r', label='fitted line')
ax.set_xlabel('channels linear location')
ax.set_ylabel('points linear location')
ax.legend()
plt.show()

print(res)

# %%
np.mean(np.abs(errors))

# %%
1- res.rvalue**2

# %% [markdown]
# Mean error should be around 75 m, maximum error around 300 m

# %% [markdown]
# ## Computing channel locations

# %%
channels_pointlinloc = res.intercept + res.slope * channels_linloc

# which channels are not predicted out of the fiber
min_pointlinloc = points_linloc[0]
max_pointlinloc = points_linloc[-1]
channels_isvalid = (channels_pointlinloc > min_pointlinloc) & (channels_pointlinloc < max_pointlinloc)
valid_channels = channels[channels_isvalid]

# locating channels
points_lonlat = np.array(points_gdf.geometry.get_coordinates())
points_wind = wind_crs.transform_points(lonlat_crs, points_lonlat[:, 0], points_lonlat[:, 1])
channels_wind_x = np.interp(channels_pointlinloc, points_linloc, points_wind[:, 0])
channels_wind_y = np.interp(channels_pointlinloc, points_linloc, points_wind[:, 1])
channels_lonlat = lonlat_crs.transform_points(wind_crs, channels_wind_x, channels_wind_y)[:,:2]

# adding information relative to the closest osm point
channels_pointindex = np.interp(channels_pointlinloc, points_linloc, np.arange(len(points_linloc)))
channels_pointindex = np.round(channels_pointindex).astype(int)
channels_pointlon = points_lonlat[channels_pointindex, 0]
channels_pointlat = points_lonlat[channels_pointindex, 1]
channels_power = fiber_points_df.loc[channels_pointindex, "power"].values
channels_location = fiber_points_df.loc[channels_pointindex, "location"].values

channels_df = pd.DataFrame({
    "channel": channels,
    "valid": channels_isvalid,
    "lon": channels_lonlat[:, 0],
    "lat": channels_lonlat[:, 1],
    "wind_crs_x": channels_wind_x,
    "wind_crs_y": channels_wind_y,
    "linloc": channels_linloc,
    "point_linloc": channels_pointlinloc,
    "point_index": channels_pointindex,
    "point_lon": channels_pointlon,
    "point_lat": channels_pointlat,
    "power": channels_power,
    "location": channels_location
})

# uncomment to save
#channels_df.to_csv(path_dict["channels_osm"], index=False)
channels_df.head()

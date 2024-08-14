import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import io
from urllib.request import urlopen, Request
from PIL import Image


tile_providers = {
    "OSM": cimgt.OSM,
    "GoogleTiles": cimgt.GoogleTiles,
    "GoogleWTS": cimgt.GoogleWTS,
    "MapQuestOSM": cimgt.MapQuestOSM,
    "MapQuestOpenAerial": cimgt.MapQuestOpenAerial,
    "MapboxStyleTiles": cimgt.MapboxStyleTiles,
    "MapboxTiles": cimgt.MapboxTiles,
    "OrdnanceSurvey": cimgt.OrdnanceSurvey,
    "QuadtreeTiles": cimgt.QuadtreeTiles,
    "StadiaMapsTiles": cimgt.StadiaMapsTiles,
    "Stamen": cimgt.Stamen,
}


working_tile_provider_names = [
    "OSM",
    "GoogleTiles",
    "QuadtreeTiles",
]


default_extents = {
    "world": (-180, 180, -90, 90),
    "europe": (-12, 42, 35, 72),
    "europe_north": (-5, 28, 45, 72),
    "norway": (4, 32, 57, 71),
    "lofoten": (12, 16, 67, 69),
    "lofoten_das": (13, 13.8, 67.95, 68.22),
    "lofoten_das_tight": (13.1, 13.7, 67.97, 68.2),
    "leknes_lufthavn": (13.56, 13.63, 68.145, 68.17),
    "kakern_bru": (13.15, 13.2, 68.005, 68.03),
}


def _image_spoof(self, tile): 

    r"""helper function to download map tiles for cartopy
    
    comes from https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy"""
    
    # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy


def add_map_old(
        ax: plt.Axes,
        extent: tuple[float, float, float, float]
    ):

    r"""Add geo features to a matplotlib axes: map, grid, extent

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to add the map to.
    extent : tuple[float, float, float, float]
        The extent of the map to show in the form (lon_min, lon_max, lat_min, lat_max).
    """
    
    # unpack extent
    longitude_min, longitude_max, latitude_min, latitude_max = extent

    ax.gridlines(draw_labels=["bottom", "left"], x_inline=False, y_inline=False)
    ax.set_extent([longitude_min, longitude_max, latitude_min, latitude_max], crs=ccrs.PlateCarree())

    cimgt.GoogleTiles.get_image = _image_spoof # reformat web request for street map spoofing
    map_img = cimgt.GoogleTiles(cache=True) # spoofed, downloaded street map
    zoom = 0.5 * np.max([longitude_max - longitude_min, latitude_max - latitude_min])
    scale = np.ceil( - np.sqrt(2) * np.log( np.divide(zoom,350.0) ) ) # empirical solve for scale based on zoom
    scale = (scale < 20) and scale or 19 # scale cannot be larger than 19
    ax.add_image(map_img, int(scale), alpha=0.5) # add map_img with zoom specification  


def add_map(
        ax: plt.Axes,
        extent: tuple[float, float, float, float],
        tile_provider: str = "QuadtreeTiles",
        alpha: float = 0.5, 
        scale_modifier: int = 0, 
        draw_labels=["bottom", "left"]
    ):

    r"""Add geo features to a matplotlib axes: map, grid, extent

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to add the map to.
    extent : tuple[float, float, float, float]
        The extent of the map to show in the form (lon_min, lon_max, lat_min, lat_max).
    tile_provider : str, optional
        The tile provider to use. see cartopy.io.img_tiles for possible values.
        Default is "QuadtreeTiles".
    alpha : float, optional
        The transparency of the map. Default is 0.5.
    """
    
    # unpack extent
    longitude_min, longitude_max, latitude_min, latitude_max = extent

    ax.gridlines(draw_labels=draw_labels, x_inline=False, y_inline=False)
    ax.set_extent([longitude_min, longitude_max, latitude_min, latitude_max], crs=ccrs.PlateCarree())

    Tiles = tile_providers[tile_provider]
    map_img = Tiles(cache=True) # spoofed, downloaded street map
    Tiles.get_image = _image_spoof # reformat web request for street map spoofing
    zoom = 0.5 * np.max([longitude_max - longitude_min, latitude_max - latitude_min])
    scale = np.ceil( - np.sqrt(2) * np.log( np.divide(zoom,350.0) ) ) # empirical solve for scale based on zoom
    scale += scale_modifier
    scale = (scale < 20) and scale or 19 # scale cannot be larger than 19
    ax.add_image(map_img, int(scale), alpha=alpha) # add map_img with zoom specification

"""
Some utility functions that are shared across checks
"""
import geojson
import numpy as np
from affine import Affine
from scipy.ndimage import find_objects, maximum_filter
from osgeo import gdal, ogr, osr

from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.mbesgc.lib.data import InputFileDetails

def remove_edge_labels(labeled_array: np.ndarray) -> np.ndarray:
    """ Removes labels from the input labeled_array that have at least
    one element that touches an edge. Operates in place.
    """
    # get all the unique patch ids from the edges of the labeled array
    top = labeled_array[0]
    bottom = labeled_array[labeled_array.shape[0] - 1]
    left = labeled_array[:, 0]
    right = labeled_array[:, labeled_array.shape[1] - 1]

    # we only want one array that contains the unique patch ids that
    # touch the edge of array
    all_edges = np.concatenate((top, bottom, left, right))
    all_edges_unique = np.unique(all_edges)
    # remove 0 as these pixels have data (and aren't holes)
    all_edges_unique = all_edges_unique[all_edges_unique != 0]

    # get the location of all patches
    object_slices = find_objects(labeled_array)
    for obj_slice in object_slices:
        # get the patch data
        patch = labeled_array[obj_slice]
        # now replace the only patch data that is a hole that touches an
        # edge. It's possible for a hole that touches an edge to suround
        # a hole that doesn't touch the edge. eg; the 2 below
        # [1 0 2 0 1 0 ]
        # [1 0 0 0 1 0 ]
        # [1 1 1 1 1 0 ]
        # [0 0 0 0 0 0 ]
        patch = np.where(
            np.isin(
                patch,
                all_edges_unique,
                assume_unique=False),
            0,
            patch
        )
        # now replace the data in the labeled array with the updated
        # patch
        labeled_array[obj_slice] = patch
    
    return labeled_array


def grow_pixels(data_array: np.ndarray, pixel_growth: int) -> np.ndarray:
    '''
    Used for boolean data arrays, will grow out a non-zero (true) pixel
    value by a certain number of pixels. Helps fatten up areas that fail
    a check and supports more simple ploygonised geometry.
    '''
    return maximum_filter(
        data_array,
        size=(pixel_growth, pixel_growth)
    )


def __add_geom(geom, out_lyr):
    feature_def = out_lyr.GetLayerDefn()
    out_feat = ogr.Feature(feature_def)
    out_feat.SetGeometry(geom)
    out_lyr.CreateFeature(out_feat)


def simplify_layer(in_lyr, out_lyr, simplify_distance):
    '''
    Creates a simplified layer from an input layer using GDAL's
    simplify function
    '''
    for in_feat in in_lyr:
        geom = in_feat.GetGeometryRef()
        simple_geom = geom.SimplifyPreserveTopology(simplify_distance)
        __add_geom(simple_geom, out_lyr)


def labeled_array_to_geojson(
        labeled_array: np.ndarray,
        tile: Tile,
        ifd: InputFileDetails,
        pixel_growth: int
    ) -> list[geojson.Feature]:
    """
    """
    labeled_array = labeled_array.astype(np.int16)

    tile_ds = gdal.GetDriverByName('MEM').Create(
        '',
        tile.max_x - tile.min_x,
        tile.max_y - tile.min_y,
        1,
        gdal.GDT_CInt16
    )

    # grow out failed pixels to make them more obvious. We've already
    # calculated the pass/fail stats so this won't impact results.
    labeled_array_grow = grow_pixels(labeled_array, pixel_growth)

    # simplify distance is calculated as the distance pixels are grown out
    # `ifd.geotransform[1]` is pixel size
    simplify_distance = 0.5 * pixel_growth * ifd.geotransform[1]

    src_affine = Affine.from_gdal(*ifd.geotransform)
    tile_affine = src_affine * Affine.translation(
        tile.min_x,
        tile.min_y
    )

    tile_ds.SetGeoTransform(tile_affine.to_gdal())

    tile_band = tile_ds.GetRasterBand(1)
    tile_band.WriteArray(labeled_array_grow, 0, 0)
    tile_band.SetNoDataValue(0)
    tile_band.FlushCache()
    tile_ds.SetProjection(ifd.projection)

    ogr_srs = osr.SpatialReference()
    ogr_srs.ImportFromWkt(ifd.projection)

    ogr_driver = ogr.GetDriverByName('Memory')
    ogr_dataset = ogr_driver.CreateDataSource('shapemask')
    ogr_layer = ogr_dataset.CreateLayer('shapemask', srs=ogr_srs)

    # used the input raster data 'tile_band' as the input and mask, if not
    # used as a mask then a feature that outlines the entire dataset is
    # also produced
    gdal.Polygonize(
        tile_band,
        tile_band,
        ogr_layer,
        -1,
        [],
        callback=None
    )

    ogr_simple_driver = ogr.GetDriverByName('Memory')
    ogr_simple_dataset = ogr_simple_driver.CreateDataSource(
        'failed_poly')
    ogr_simple_layer = ogr_simple_dataset.CreateLayer(
        'failed_poly', srs=None)

    simplify_layer(
        ogr_layer,
        ogr_simple_layer,
        simplify_distance)

    ogr_srs_out = osr.SpatialReference()
    ogr_srs_out.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

    features = []
    for feature in ogr_simple_layer:
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform)
        geojson_feature = geojson.loads(feature.ExportToJson())
        print(type(geojson_feature))
        print(geojson_feature)
        features.append(geojson_feature)

    ogr_simple_dataset.Destroy()
    ogr_dataset.Destroy()

    return features

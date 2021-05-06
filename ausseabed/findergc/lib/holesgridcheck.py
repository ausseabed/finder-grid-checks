from affine import Affine
from osgeo import gdal, ogr, osr
from scipy.ndimage import find_objects, label
from typing import Optional, Dict, List, Any
import collections
import geojson
from geojson import MultiPolygon
import numpy as np
import numpy.ma as ma

from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.gridcheck import GridCheck, GridCheckState, \
    GridCheckResult
from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution


class HolesCheck(GridCheck):
    '''
    Checks for holes in data as indicated by patches of nodata
    '''

    id = 'e80b365f-23b6-4bea-9a24-1a40b3e46c1e'
    name = 'Hole Finder Check'
    version = '1'

    input_params = [
        QajsonParam("Ignore edge holes", True)
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self.ignore_edges = self.get_param('Ignore edge holes')

        self.tiles_geojson = MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth = 5

    def merge_results(self, last_check: GridCheck):
        self.start_time = last_check.start_time

        self.hole_count += last_check.hole_count
        self.hole_pixels += last_check.hole_pixels
        self.total_cell_count += last_check.total_cell_count

        self.tiles_geojson.coordinates.extend(
            last_check.tiles_geojson.coordinates
        )

        self._merge_temp_dirs(last_check)

    def run(
            self,
            ifd: InputFileDetails,
            tile: Tile,
            depth,
            density,
            uncertainty,
            progress_callback=None):
        # run check on tile data
        self.total_cell_count = int(depth.count())
        self.hole_count = 0
        self.hole_pixels = 0

        if self.total_cell_count == 0:
            return

        # the mask tells us what pixels are nodata (could be holes)
        mask = np.ma.getmask(depth)
        # define a structure that will consider diagonal links
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        # use label to uniquely identify each contiguous patch of nodata, this
        # gives each patch a unique id
        labeled_array, _ = label(mask, structure=s, output=np.int32)

        if self.ignore_edges:
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

        unique_vals, unique_counts = np.unique(
            labeled_array, return_counts=True)
        hist = {}
        for (val, count) in zip(unique_vals, unique_counts):
            if isinstance(val, ma.core.MaskedConstant):
                continue
            hist[int(val)] = int(count)

        # remove 0, this is actual data not a hole
        del hist[0]

        for _, hole_px_count in hist.items():
            self.hole_count += 1
            self.hole_pixels += hole_px_count

        if not (self.spatial_export or self.spatial_export_location):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(
            tile.min_x,
            tile.min_y
        )

        if self.spatial_qajson:

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
            labeled_array_grow = self._grow_pixels(
                labeled_array, self.pixel_growth)

            # simplify distance is calculated as the distance pixels are grown out
            # `ifd.geotransform[1]` is pixel size
            simplify_distance = 0.5 * self.pixel_growth * ifd.geotransform[1]

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

            self._simplify_layer(
                ogr_layer,
                ogr_simple_layer,
                simplify_distance)

            ogr_srs_out = osr.SpatialReference()
            ogr_srs_out.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

            for feature in ogr_simple_layer:
                transformed = feature.GetGeometryRef()
                transformed.Transform(transform)
                geojson_feature = geojson.loads(feature.ExportToJson())
                self.tiles_geojson.coordinates.extend(
                    geojson_feature.geometry.coordinates
                )

            ogr_simple_dataset.Destroy()
            ogr_dataset.Destroy()

        if self.spatial_export:
            tf = self._get_tmp_file('holes', 'tif', tile)
            tile_ds = gdal.GetDriverByName('GTiff').Create(
                tf,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Int16,
                options=['COMPRESS=DEFLATE']
            )

            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(labeled_array, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            ogr_srs = osr.SpatialReference()
            ogr_srs.ImportFromWkt(ifd.projection)

            sf = self._get_tmp_file('holes', 'shp', tile)
            ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
            ogr_dataset = ogr_driver.CreateDataSource(sf)
            ogr_layer = ogr_dataset.CreateLayer('holes', srs=ogr_srs)

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

            tile_ds = None
            ogr_dataset.Destroy()

            self._move_tmp_dir()

    def __get_messages_from_data(self, data, total_cells, total_failed_cells):
        ''' Generates a human readable summary of the data dict that is
        populated with failed check counts.
        '''
        start_str = 'failed_cell_'
        messages = []
        pc_fail = total_failed_cells / total_cells * 100.0
        messages.append(
            f"{total_failed_cells} nodes failed the flier finders check, this "
            f"represents {pc_fail:.1f}% of all nodes."
        )

        return messages

    def get_outputs(self) -> QajsonOutputs:

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message
        )

        data = {
            "total_hole_count": self.hole_count,
            "total_hole_cell_count": self.hole_pixels,
            "total_cell_count": self.total_cell_count
        }

        if self.hole_count > 0:
            check_state = GridCheckState.cs_fail
            if self.spatial_qajson:
                data['map'] = self.tiles_geojson

            msg = (
                f"A total of {self.hole_count} holes were found. The total area "
                f"of these holes was {self.hole_pixels}px.")
            messages = [msg]
        else:
            check_state = GridCheckState.cs_pass
            messages = ["No holes found"]

        return QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=messages,
            data=data,
            check_state=check_state
        )

from affine import Affine
from osgeo import gdal, ogr, osr
from scipy.ndimage import find_objects, label
from time import perf_counter
from typing import Optional, Dict, List, Any
import collections
import geojson
from geojson import MultiPolygon
import logging
import numpy as np
import numpy.ma as ma

from ausseabed.findergc.lib.utils import remove_edge_labels, labeled_array_to_geojson
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.gridcheck import GridCheck, GridCheckState, \
    GridCheckResult
from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution

logger = logging.getLogger(__name__)


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
        self.extents_geojson = geojson.MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth = 5

        self.missing_depth = None

    def merge_results(self, last_check: GridCheck):
        self.start_time = last_check.start_time

        if self.execution_status == "aborted" or self.execution_status == "failed":
            return

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
            pinkchart,
            progress_callback=None):
        # run check on tile data

        t1_start = perf_counter()

        # this check only requires the depth layer, so check it is given
        # if not mark this check as aborted
        self.missing_depth = depth is None
        if self.missing_depth:
            self.execution_status = "aborted"
            self.error_message = "Missing depth data"
            logger.info(f"{self.error_message}, aborting hole finder check")
            # we cant run the check so return
            return

        # if there's no pink chart data then use the number of non-nodata cells
        # as the cell count. Otherwise count the number of pinkchart cells
        self.total_cell_count = int(depth.count())
        if pinkchart is not None:
            self.total_cell_count = int(pinkchart.sum())
            # turn off ignore edges if there is a pink chart specified
            # the method implemented below does not incorperate the pink chart
            # layer, and it's very unlikely the use would want to ignore edge
            # holes if they have specified a pink chart layer.
            self.ignore_edges = False

        self.hole_count = 0
        self.hole_pixels = 0

        if self.total_cell_count == 0:
            return

        # the mask tells us what pixels are nodata (could be holes)
        # use getmaskarray in place of getmask as getmask will return an
        # empty dimensionless array when there is no nodata in the depth
        # array.
        mask = np.ma.getmaskarray(depth)

        if pinkchart is not None:
            mask = (mask & pinkchart)

        # define a structure that will consider diagonal links
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        # use label to uniquely identify each contiguous patch of nodata, this
        # gives each patch a unique id
        labeled_array, _ = label(mask, structure=s, output=np.int32)

        if self.ignore_edges:
            remove_edge_labels(labeled_array)

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

        t1_stop = perf_counter()
        logger.debug(f"Holes check time = {t1_stop - t1_start:.4f}s")

        if not (self.spatial_export or self.spatial_export_location or self.spatial_qajson):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(
            tile.min_x,
            tile.min_y
        )

        if self.spatial_qajson:
            spatial_qajson_start = perf_counter()

            self.extents_geojson = ifd.get_extents_feature()

            features = labeled_array_to_geojson(
                labeled_array,
                tile,
                ifd,
                self.pixel_growth
            )

            for feature in features:
                self.tiles_geojson.coordinates.append(
                    feature.geometry.coordinates
                )

            spatial_qajson_stop = perf_counter()
            logger.debug(f"Holes spatial QAJSON time = {spatial_qajson_stop - spatial_qajson_start:.4f}s")

        if self.spatial_export:
            spatial_detailed_start = perf_counter()
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

            spatial_detailed_stop = perf_counter()
            logger.debug(f"Holes raster export time = {spatial_detailed_stop - spatial_detailed_start:.4f}s")

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

        messages = []
        data = {}
        if self.execution_status == "completed":
            data = {
                "total_hole_count": self.hole_count,
                "total_hole_cell_count": self.hole_pixels,
                "total_cell_count": self.total_cell_count
            }

        if self.execution_status == "aborted" or self.execution_status == "failed":
            check_state = GridCheckState.cs_fail
        elif self.hole_count > 0:
            check_state = GridCheckState.cs_fail
            if self.spatial_qajson:
                data['map'] = self.tiles_geojson
                data['extents'] = self.extents_geojson

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

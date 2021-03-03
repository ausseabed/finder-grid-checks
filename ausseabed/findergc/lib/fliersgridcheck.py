from affine import Affine
from osgeo import gdal, ogr, osr
from scipy import ndimage
from typing import Optional, Dict, List, Any
import collections
import geojson
import numpy as np
import numpy.ma as ma

from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.gridcheck import GridCheck, GridCheckState, \
    GridCheckResult
from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution

from ausseabed.findergc.lib import fliers


class FliersCheck(GridCheck):
    '''

    '''

    id = '04519264-00ec-4e48-8abc-23cdf3c4913e'
    name = 'Flier Finder Check'
    version = '1'

    # default values taken from IHO - 1a spec
    input_params = [
        QajsonParam("Laplacian Operator - threshold", 1.0),
        QajsonParam("Noisy Edges - dist", 2),
        QajsonParam("Noisy Edges - cf", 1.0),
        QajsonParam("Adjacent Cells - threshold", 2.0),
        QajsonParam("Adjacent Cells - percent 1", 20.0),
        QajsonParam("Adjacent Cells - percent 2", 20.0),
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self.laplace_threshold = self.get_param(
            'Laplacian Operator - threshold')

        self._ne_dist = self.get_param('Noisy Edges - dist')
        self._ne_cf = self.get_param('Noisy Edges - cf')

        self._ac_threshold = self.get_param('Adjacent Cells - threshold')
        self._ac_pc1 = self.get_param('Adjacent Cells - percent 1')
        self._ac_pc2 = self.get_param('Adjacent Cells - percent 2')

        self.geojson_points = []

    def merge_results(self, last_check: GridCheck):
        self.start_time = last_check.start_time

        self.total_cell_count += last_check.total_cell_count
        self.failed_cell_laplacian_operator += self.failed_cell_laplacian_operator
        self.failed_cell_count_noisy_edges += last_check.failed_cell_count_noisy_edges
        self.failed_cell_adjacent_cells += last_check.failed_cell_adjacent_cells

        self.geojson_points.extend(last_check.geojson_points)

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

        flag_grid = np.full(
            depth.shape,
            0,
            dtype=np.int
        )

        depth_clone = depth.copy()

        depth_clone.fill_value = np.NaN
        depth_clone = depth_clone.filled()

        # run the laplacian operator check
        depth_laplace = ndimage.filters.laplace(depth_clone)
        fliers.laplacian_operator(
            depth_laplace,
            flag_grid,
            threshold=self.laplace_threshold
        )

        # run adjacent cells check
        fliers.adjacent_cells(
            depth_clone,
            flag_grid,
            threshold=self._ac_threshold,
            percent_1=self._ac_pc1,
            percent_2=self._ac_pc2
        )

        # run noisy edges check
        fliers.noisy_edges(
            depth_clone,
            flag_grid,
            dist=self._ne_dist,
            cf=self._ne_cf
        )

        # tf = '/Users/lachlan/work/projects/qa4mb/repo/finder-grid-checks/au2.tif'
        # tile_ds = gdal.GetDriverByName('GTiff').Create(
        #     tf,
        #     tile.max_x - tile.min_x,
        #     tile.max_y - tile.min_y,
        #     1,
        #     gdal.GDT_Float32
        # )
        # src_affine = Affine.from_gdal(*ifd.geotransform)
        # tile_affine = src_affine * Affine.translation(
        #     tile.min_x,
        #     tile.min_y
        # )
        # tile_ds.SetGeoTransform(tile_affine.to_gdal())
        #
        # tile_band = tile_ds.GetRasterBand(1)
        # tile_band.WriteArray(depth_laplace, 0, 0)
        # tile_band.SetNoDataValue(0)
        # tile_band.FlushCache()
        # tile_ds.SetProjection(ifd.projection)


        # laplacian_operator check uses 1 as its flag value
        self.failed_cell_laplacian_operator = np.count_nonzero(flag_grid == 1)
        # adjacent cells uses 6 as its flag value
        self.failed_cell_adjacent_cells = np.count_nonzero(flag_grid == 3)
        # noisy edges uses 6 as its flag value
        self.failed_cell_count_noisy_edges = np.count_nonzero(flag_grid == 6)

        # use the details of the raster's geotransform to translate from
        # pixel coords to the rasters coord system
        origin_x = ifd.geotransform[0]
        origin_y = ifd.geotransform[3]
        pixel_width = ifd.geotransform[1]
        pixel_height = ifd.geotransform[5]

        src_proj = osr.SpatialReference()
        src_proj.ImportFromWkt(ifd.projection)
        dst_proj = osr.SpatialReference()
        dst_proj.ImportFromEPSG(4326)
        coord_trans = osr.CoordinateTransformation(src_proj, dst_proj)

        self.geojson_points = []

        # get locations of all nodes that have failed one of the flier checks
        failed_cell_indicies = np.argwhere(flag_grid > 0)
        # iterate through each one and create a point feature
        for row, col in failed_cell_indicies:
            flag = flag_grid[row][col]

            x = origin_x + pixel_width * col
            y = origin_y + pixel_height * row

            # now reproject into 4326 as required by geojson
            x, y = coord_trans.TransformPoint(x, y, 0.0)[:2]

            pt = geojson.Point([x, y])
            feature = geojson.Feature(
                geometry=pt,
                properties={
                    'flag': int(flag)
                }
            )
            self.geojson_points.append(feature)

    def get_outputs(self) -> QajsonOutputs:

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message
        )

        data = {
            "failed_cell_laplacian_operator": self.failed_cell_laplacian_operator,
            "failed_cell_count_noisy_edges": self.failed_cell_count_noisy_edges,
            "failed_cell_adjacent_cells": self.failed_cell_adjacent_cells,
            "total_cell_count": self.total_cell_count
        }

        map_feature = geojson.FeatureCollection(self.geojson_points)
        map = geojson.mapping.to_mapping(map_feature)

        data['map'] = map

        return QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=[],
            data=data,
            check_state=GridCheckState.cs_pass
        )

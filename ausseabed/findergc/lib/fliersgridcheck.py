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
        QajsonParam("Gaussian Curvature - threshold", 1.0),
        QajsonParam("Noisy Edges - dist", 2.0),
        QajsonParam("Noisy Edges - cf", 1.0),
        QajsonParam("Adjacent Cells - threshold", 2.0),
        QajsonParam("Adjacent Cells - percent 1", 20.0),
        QajsonParam("Adjacent Cells - percent 2", 20.0),
        QajsonParam("Small Groups - threshold", 1.0),
        QajsonParam("Small Groups - area limit", 1.0),
        QajsonParam("Small Groups - check slivers", True),
        QajsonParam("Small Groups - check isolated", True)
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self.laplace_threshold = self.get_param(
            'Laplacian Operator - threshold')
        self.gaussian_threshold = self.get_param(
            'Gaussian Curvature - threshold')

        self._ne_dist = self.get_param('Noisy Edges - dist')
        self._ne_cf = self.get_param('Noisy Edges - cf')

        self._ac_threshold = self.get_param('Adjacent Cells - threshold')
        self._ac_pc1 = self.get_param('Adjacent Cells - percent 1')
        self._ac_pc2 = self.get_param('Adjacent Cells - percent 2')

        self._sg_threshold = self.get_param('Small Groups - threshold')
        self._sg_area_limit = self.get_param('Small Groups - area limit')
        self._sg_check_slivers = self.get_param('Small Groups - check slivers')
        self._sg_check_isolated = self.get_param(
            'Small Groups - check isolated')

        # maximium number of points that will be included in the spatial outputs
        # doesn't change reported stats, only what is diplayed in map widget
        # per tile number, so total number across all tiles can be much larger
        self.max_geojson_points = 1000
        self.max_geojson_points_exceeded = False
        self.geojson_points = []

        self.extents_geojson = geojson.MultiPolygon()

        self.missing_depth = None

    def merge_results(self, last_check: GridCheck):
        self.start_time = last_check.start_time

        if self.execution_status == "aborted":
            return

        self.total_cell_count += last_check.total_cell_count
        self.failed_cell_laplacian_operator += last_check.failed_cell_laplacian_operator
        self.failed_cell_gaussian_curvature += last_check.failed_cell_gaussian_curvature
        self.failed_cell_count_noisy_edges += last_check.failed_cell_count_noisy_edges
        self.failed_cell_adjacent_cells += last_check.failed_cell_adjacent_cells
        self.failed_cell_isolated_group += last_check.failed_cell_isolated_group
        self.failed_cell_sliver += last_check.failed_cell_sliver

        self.geojson_points.extend(last_check.geojson_points)

        self.max_geojson_points_exceeded = (
            self.max_geojson_points_exceeded or last_check.max_geojson_points_exceeded)

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

        # this check only requires the depth layer, so check it is given
        # if not mark this check as aborted
        self.missing_depth = depth is None
        if self.missing_depth:
            self.execution_status = "aborted"
            self.error_message = "Missing depth data"
            # we cant run the check so return
            return

        self.max_geojson_points_exceeded = False
        self.total_cell_count = int(depth.count())

        if self.total_cell_count == 0:
            self.failed_cell_laplacian_operator = 0
            self.failed_cell_gaussian_curvature = 0
            self.failed_cell_count_noisy_edges = 0
            self.failed_cell_adjacent_cells = 0
            self.failed_cell_isolated_group = 0
            self.failed_cell_sliver = 0
            return

        flag_grid = np.full(
            depth.shape,
            0,
            dtype=np.int16
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

        # run the gaussian curvature check
        # following lines adapted from QC Tools
        dtm_mask = np.ma.masked_invalid(depth_clone)
        gy, gx = np.gradient(dtm_mask)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        gauss_curv = (gxx * gyy - (gxy ** 2)) / \
            (1 + (gx ** 2) + (gy ** 2)) ** 2

        fliers.gaussian_curvature(
            gauss_curv,
            flag_grid,
            threshold=self.gaussian_threshold
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

        # run small groups check
        grid_bin = np.isfinite(depth_clone)
        fliers.small_groups(
            grid_bin=grid_bin,
            bathy=depth_clone,
            flag_grid=flag_grid,
            th=self._sg_threshold,
            area_limit=self._sg_area_limit,
            check_slivers=self._sg_check_slivers,
            check_isolated=self._sg_check_isolated
        )

        # tf = f"Z:/work/projects/qa4mb/repo/finder-grid-checks/lap_{tile.min_x}_{tile.min_y}.tif"
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
        # tile_band.WriteArray(np.abs(depth_laplace), 0, 0)
        # tile_band.SetNoDataValue(0)
        # tile_band.FlushCache()
        # tile_ds.SetProjection(ifd.projection)

        # laplacian_operator check uses 1 as its flag value
        self.failed_cell_laplacian_operator = np.count_nonzero(flag_grid == 1)
        # gaussian_curvature check uses 2 as its flag value
        self.failed_cell_gaussian_curvature = np.count_nonzero(flag_grid == 2)
        # adjacent cells uses 3 as its flag value
        self.failed_cell_adjacent_cells = np.count_nonzero(flag_grid == 3)
        if self._sg_check_slivers:
            # slivers uses 4 as its flag value
            self.failed_cell_sliver = np.count_nonzero(flag_grid == 4)
        if self._sg_check_isolated:
            # isolated group uses 5 as its flag value
            self.failed_cell_isolated_group = np.count_nonzero(flag_grid == 5)
        # noisy edges uses 6 as its flag value
        self.failed_cell_count_noisy_edges = np.count_nonzero(flag_grid == 6)

        if not (self.spatial_export or self.spatial_export_location):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(
            tile.min_x,
            tile.min_y
        )
        geotransform = tile_affine.to_gdal()

        # use the details of the raster's geotransform to translate from
        # pixel coords to the rasters coord system
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        src_proj = osr.SpatialReference()
        src_proj.ImportFromWkt(ifd.projection)
        dst_proj = osr.SpatialReference()
        dst_proj.ImportFromEPSG(4326)
        dst_proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        coord_trans = osr.CoordinateTransformation(src_proj, dst_proj)

        self.geojson_points = []

        if self.spatial_qajson:
            self.extents_geojson = ifd.get_extents_feature()

            # get locations of all nodes that have failed one of the flier
            # checks
            failed_cell_indicies = np.argwhere(flag_grid > 0)

            failed_point_count = 0
            # iterate through each one and create a point feature
            for row, col in failed_cell_indicies:
                if failed_point_count > self.max_geojson_points:
                    # qajson gets large if too many points are included in the output.
                    # So limit the maximum anount of points. This doesnt effect the
                    # reported stats, only what is shown in the map widget.
                    self.max_geojson_points_exceeded = True
                    break
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

                failed_point_count += 1

        if self.spatial_export:

            lap_f = self._get_tmp_file('laplace', 'tif', tile)
            lap_tile_ds = gdal.GetDriverByName('GTiff').Create(
                lap_f,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=DEFLATE']
            )

            lap_tile_ds.SetGeoTransform(tile_affine.to_gdal())

            lap_tile_band = lap_tile_ds.GetRasterBand(1)
            lap_tile_band.WriteArray(depth_laplace, 0, 0)
            lap_tile_band.SetNoDataValue(0)
            lap_tile_band.FlushCache()
            lap_tile_ds.SetProjection(ifd.projection)
            lap_tile_ds = None

            gauss_f = self._get_tmp_file('gaussian', 'tif', tile)
            gauss_tile_ds = gdal.GetDriverByName('GTiff').Create(
                gauss_f,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=DEFLATE']
            )

            gauss_tile_ds.SetGeoTransform(tile_affine.to_gdal())

            gauss_tile_band = gauss_tile_ds.GetRasterBand(1)
            gauss_tile_band.WriteArray(gauss_curv, 0, 0)
            gauss_tile_band.SetNoDataValue(0)
            gauss_tile_band.FlushCache()
            gauss_tile_ds.SetProjection(ifd.projection)
            gauss_tile_ds = None

            tf = self._get_tmp_file('fliers', 'tif', tile)
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
            tile_band.WriteArray(flag_grid, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)
            tile_ds = None

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
        for key, value in data.items():
            if key.startswith(start_str) and value > 0:
                pc_fail_loc = value / total_cells * 100.0
                quant = key[len(start_str):].replace('_', ' ')
                str = f"   {value} nodes failed {quant} ({pc_fail_loc:.2f}%)"
                messages.append(str)

        return messages

    def get_outputs(self) -> QajsonOutputs:

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message
        )

        data = {}
        if self.execution_status != "aborted":
            data = {
                "failed_cell_laplacian_operator": self.failed_cell_laplacian_operator,
                "failed_cell_gaussian_curvature": self.failed_cell_gaussian_curvature,
                "failed_cell_count_noisy_edges": self.failed_cell_count_noisy_edges,
                "failed_cell_adjacent_cells": self.failed_cell_adjacent_cells,
                "total_cell_count": self.total_cell_count
            }

            total_failed = sum([
                self.failed_cell_laplacian_operator,
                self.failed_cell_gaussian_curvature,
                self.failed_cell_count_noisy_edges,
                self.failed_cell_adjacent_cells
            ])

            if self._sg_check_slivers:
                data["failed_cell_sliver"] = self.failed_cell_sliver
                total_failed += self.failed_cell_sliver
            if self._sg_check_isolated:
                data["failed_cell_isolated_group"] = self.failed_cell_isolated_group
                total_failed += self.failed_cell_isolated_group

        if self.execution_status == "aborted":
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[self.error_message],
                data=data,
                check_state=GridCheckState.cs_fail
            )
        elif total_failed > 0:
            check_state = GridCheckState.cs_fail
            if self.spatial_qajson:
                map_feature = geojson.FeatureCollection(self.geojson_points)
                map = geojson.mapping.to_mapping(map_feature)
                data['map'] = map
                data['extents'] = self.extents_geojson

            messages = self.__get_messages_from_data(
                data,
                self.total_cell_count,
                total_failed
            )
        else:
            check_state = GridCheckState.cs_pass
            messages = []

        if self.max_geojson_points_exceeded:
            messages.append(
                "Warning: maximum number of viewable points was "
                "exceeded. Not all failed points will be displayed "
                "in the map view and qajson.")

        return QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=messages,
            data=data,
            check_state=check_state
        )

import collections
import geojson
import logging
import numpy as np
from osgeo import gdal
from scipy.ndimage import label, convolve, maximum
from numpy.typing import ArrayLike
from time import perf_counter
from typing import List

from ausseabed.findergc.lib.utils import  \
    remove_edge_labels, labeled_array_to_geojson, save_raster, save_raster_as_vector
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.gridcheck import GridCheck, GridCheckState
from ausseabed.mbesgc.lib.tiling import Tile


logger = logging.getLogger(__name__)


class HoleAndGapCheck(GridCheck):
    '''
    Checks for data holes and data gaps as indicated by patches of nodata
    '''

    id = 'd095f584-a4aa-4f92-b123-336e1e328289'
    name = 'Data Hole and Data Gap Check'
    version = '1'

    # no input params required for this check
    input_params = [
        QajsonParam("Ignore edge holes", True),
        QajsonParam("Minimum soundings per node", 5),
        QajsonParam("Gap area threshold (%)", 2),
        QajsonParam("Hole area threshold (%)", 0),
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self.ignore_edges = self.get_param('Ignore edge holes')
        self.density_threshold = self.get_param('Minimum soundings per node')
        self.gap_area_threshold = self.get_param('Gap area threshold (%)') / 100
        self.hole_area_threshold = self.get_param('Hole area threshold (%)') / 100

        # initialise the output geojson to empty geom
        self.tiles_geojson = geojson.MultiPolygon()
        self.extents_geojson = geojson.MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth = 5

    def merge_results(self, last_check: GridCheck):
        # technically QAX processes data through this check in tiles. It's this
        # function that is called after all tiles have been processed to merge
        # the results from all tiles into a single tile (so the total results
        # are presented to the user)
        # Note: default tile size is now deliberately chosen to be so large
        # there will likely only even be one tile.
        self.start_time = last_check.start_time

        if self.execution_status == "aborted":
            return

        self.hole_count += last_check.hole_count
        self.hole_pixels += last_check.hole_pixels
        self.gap_count += last_check.gap_count
        self.gap_pixels += last_check.gap_pixels
        self.total_cell_count += last_check.total_cell_count

        self.hole_hist = self.__merge_hist(self.hole_hist, last_check.hole_hist)
        self.gap_hist = self.__merge_hist(self.hole_hist, last_check.gap_hist)

        self.tiles_geojson.coordinates.extend(
            last_check.tiles_geojson.coordinates
        )

        self._merge_temp_dirs(last_check)

    def __calc_count_and_histogram(self, arr: np.array) -> tuple[int, dict[int,  int]]:
        """ extracts the number of unique holes (or gaps) and also returns a
        histogram that details number of holes with how many cells
        """
        unique_vals, unique_counts = np.unique(arr, return_counts=True)
        unique_count = len(unique_vals)
        count_uniques = np.unique(unique_counts, return_counts=True)

        hist = {}
        for (val, count) in zip(*count_uniques):
            hist[int(val)] = int(count)

        return unique_count, hist

    def __merge_hist(self, hist_1: dict[int, int], hist_2:dict[int, int]) -> dict[int, int]:
        """ Merges the contents of two histograms together. If both histograms contain
        the same key, then the values for this key are added together.
        """
        merged: dict[int, int] = dict(hist_1)
        for (k, v) in hist_2.items():
            if k in merged:
                merged[k] += v
            else:
                merged[k] = v
        return merged

    def __hist_to_json_compatible(self, h: dict[int, int]) -> dict[str, int]:
        ordered_hist = collections.OrderedDict(h)
        res = collections.OrderedDict()
        for key, val in ordered_hist.items():
            res[str(key)] = val
        return res

    def run(
            self,
            ifd: InputFileDetails,
            tile: Tile,
            depth: ArrayLike,
            density: ArrayLike,
            uncertainty: ArrayLike,
            pinkchart: ArrayLike,
            progress_callback=None):
        # run check on tile data

        t1_start = perf_counter()

        # this check only requires the density layer, so check it is given
        # if not mark this check as aborted
        # density layer is required as the presence of holes is not indicated
        # by nodata values, but by density values below a threshold
        self.missing_density = density is None
        if self.missing_density:
            self.execution_status = "aborted"
            self.error_message = "Missing density data"
            logger.info(f"{self.error_message}, aborting hole and data gap check")
            # we cant run the check so return
            return

        if pinkchart is not None:
            if self.ignore_edges:
                logger.info(
                    "Ignore edges was set to True, but a coverage area file has been specified. "
                    "This setting will be ignored as the coverage area determines the edges of the "
                    "dataset."
                )
            self.ignore_edges = False

        # if there's no pink chart data then use the number of non-nodata cells
        # as the cell count. Otherwise count the number of pinkchart cells
        self.total_cell_count = int(density.count())
        if pinkchart is not None:
            self.total_cell_count = int(pinkchart.sum())

        # number of holes found
        self.hole_count = 0
        # number of pixels in all the holes found
        self.hole_pixels = 0
        # number of data gaps found
        self.gap_count = 0
        # number of pixels in all the data gaps found
        self.gap_pixels = 0

        # histogram details for holes and gaps
        # key is the number of cells in the hole/gaps, value
        # is the number of holes/gaps with that number of cells
        self.hole_hist: dict[int, int] = {}
        self.gap_hist: dict[int, int] = {}

        if self.total_cell_count == 0:
            return

        # hole and gap check is not based on nodata within the dataset, it's
        # based on what cells have a density lower than a given threshold.
        # Here's where we create that mask.
        mask = density < self.density_threshold

        if pinkchart is not None:
            mask = (mask & pinkchart)

        # we want to identify groups of cells that are 3x3 in size, each
        # hole must have at least one of these in them to be classified
        # a hole
        s = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        # now we run the kernel over the mask. Where there's holes we'll end
        # up with a value of 4, but the value 4 won't cover all of the hole
        c = convolve(mask.astype(np.int8), s, mode='constant', cval=0)

        # define a structure that will consider all links (adjacent and
        # diagonal) between cells
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        # use label to uniquely identify each contiguous hole, this
        # gives each hole a unique id (label)
        labels, label_count = label(mask, structure=s, output=np.int32)

        if self.ignore_edges:
            remove_edge_labels(labels)

        # find the maximum value of the convolution result for each one of the
        # labeled regions. A labeled region is a hole if this value is 9 in
        # at least one of the cells
        # we get a list of maximums where the index is the label from this function
        maximums = maximum(c, labels=labels, index=np.arange(1, label_count + 1))
        # the list is zero indexed, but our labels start at one. So we need to insert
        # a lookup table value for the 0 label into this list
        maximums = np.insert(maximums, 0, 0).astype(np.uint8)
        # replace all label ids, with the maximum value found for that label in its
        # area
        filled_labels = maximums[labels]
        # filled labels will now contain values 0 (for neither hole or gap), 1-8 for gaps, and
        # 9 for holes.

        # now simplify the numbering so that gaps == 1, and holes == 2
        gaps = ((filled_labels < 9) & (filled_labels > 0)).astype(np.int8)
        holes = (filled_labels == 9).astype(np.int8)

        # extract pixel counts
        self.gap_pixels = np.count_nonzero(gaps)
        self.hole_pixels = np.count_nonzero(holes)

        self.gap_count, self.gap_hist = self.__calc_count_and_histogram(gaps)
        self.hole_count, self.hole_hist = self.__calc_count_and_histogram(holes)

        t1_stop = perf_counter()
        logger.debug(f"Hole and Gap check time = {t1_stop - t1_start:.4f}s")

        if self.spatial_qajson:
            spatial_qajson_start = perf_counter()

            self.extents_geojson = ifd.get_extents_feature()

            features = labeled_array_to_geojson(
                gaps + holes,
                tile,
                ifd,
                self.pixel_growth
            )

            for feature in features:
                self.tiles_geojson.coordinates.append(
                    feature.geometry.coordinates
                )

            spatial_qajson_stop = perf_counter()
            logger.debug(f"Hole and Gap spatial QAJSON time = {spatial_qajson_stop - spatial_qajson_start:.4f}s")

        if self.spatial_export:
            spatial_detailed_start = perf_counter()
            tf_holes_raster = self._get_tmp_file('holes', 'tif', tile)
            holes_ds = save_raster(holes, tf_holes_raster, tile, ifd, gdal.GDT_Byte)
            tf_holes_vector = self._get_tmp_file('holes', 'shp', tile)
            save_raster_as_vector(holes_ds, tf_holes_vector, 'holes')

            tf_gaps_raster = self._get_tmp_file('gaps', 'tif', tile)
            gaps_ds = save_raster(gaps, tf_gaps_raster, tile, ifd, gdal.GDT_Byte)
            tf_gaps_vector = self._get_tmp_file('gaps', 'shp', tile)
            save_raster_as_vector(gaps_ds, tf_gaps_vector, 'gaps')
            spatial_detailed_stop = perf_counter()
            logger.debug(f"Hole and Gap raster export time = {spatial_detailed_stop - spatial_detailed_start:.4f}s")

            self._move_tmp_dir()


    def get_outputs(self) -> QajsonOutputs:
        """ Generate a QAJSON output object that includes the check results
        """

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
                "total_gap_count": self.gap_count,
                "total_gap_cell_count": self.gap_pixels,
                "total_cell_count": self.total_cell_count
            }
            if self.spatial_qajson:
                data['map'] = self.tiles_geojson
                data['extents'] = self.extents_geojson

            gap_fraction = self.gap_pixels / self.total_cell_count
            hole_fraction = self.hole_pixels / self.total_cell_count

        charts = [
            {
                'type': 'histogram',
                'data': self.__hist_to_json_compatible(self.hole_hist),
                'title': 'Hole size distribution'
            },
            {
                'type': 'histogram',
                'data': self.__hist_to_json_compatible(self.gap_hist),
                'title': 'Gap size distribution'
            },
        ]

        if self.execution_status == "aborted":
            check_state = GridCheckState.cs_fail
        elif hole_fraction > self.hole_area_threshold or gap_fraction > self.gap_area_threshold:
            check_state = GridCheckState.cs_fail
            messages = []
            if hole_fraction > self.hole_area_threshold:
                messages.append(
                    f"Percentage area identified as holes was found to be {hole_fraction*100.0:.5f}% "
                    f"({self.hole_pixels} cells), this exceeds the threshold of {self.hole_area_threshold*100}%"
                )
            if gap_fraction > self.gap_area_threshold:
                messages.append(
                    f"Percentage area identified as gaps was found to be {gap_fraction*100.0:.5f}% "
                    f"({self.gap_pixels} cells), this exceeds the threshold of {self.gap_area_threshold*100}%"
                )
            data['charts'] = charts
        else:
            check_state = GridCheckState.cs_pass
            messages = ["Total area of holes and gaps are under acceptable thresholds"]
            data['charts'] = charts

        return QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=messages,
            data=data,
            check_state=check_state
        )

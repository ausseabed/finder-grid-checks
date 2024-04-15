import geojson
import logging
import numpy as np
from scipy.ndimage import label, convolve, maximum
from numpy.typing import ArrayLike
from typing import List

from ausseabed.findergc.lib.utils import remove_edge_labels, labeled_array_to_geojson
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
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self.ignore_edges = self.get_param('Ignore edge holes')
        self.density_threshold = self.get_param('Minimum soundings per node')

        # initialise the output geojson to empty geom
        self.tiles_geojson = geojson.MultiPolygon()
        self.extents_geojson = geojson.MultiPolygon()

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

        self.tiles_geojson.coordinates.extend(
            last_check.tiles_geojson.coordinates
        )

        self._merge_temp_dirs(last_check)

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

        if self.total_cell_count == 0:
            return

        # hole and gap check is not based on nodata within the dataset, it's
        # based on what cells have a density lower than a given threshold.
        # Here's where we create that mask.
        mask = density < self.density_threshold

        if pinkchart is not None:
            mask = (mask & pinkchart)

        # we want to identify groups of cells that are 2x2 in size, each
        # hole must have at least one of these in them to be classified
        # a hole
        s = [
            [1, 1],
            [1, 1],
        ]
        # now we run the kernel over the mask. Where there's holes we'll end
        # up with a value of 4, but the value 4 won't cover all of the hole
        c = convolve(mask.astype(int), s, mode='constant', cval=0).astype(int)

        # define a structure that will consider only links that share a cell
        # edge (no diagonals).
        s = [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]]
        # use label to uniquely identify each contiguous hole, this
        # gives each hole a unique id (label)
        labels, label_count = label(mask, structure=s, output=np.int32)

        if self.ignore_edges:
            remove_edge_labels(labels)

        # find the maximum value of the convolution result for each one of the
        # labeled regions. A labeled region is a hole if this value is 4 in
        # at least one of the cells
        # we get a list of maximums where the index is the label from this function
        maximums = maximum(c, labels=labels, index=np.arange(1, label_count + 1))
        # the list is zero indexed, but our labels start at one. So we need to insert
        # a lookup table value for the 0 label into this list
        maximums = np.insert(maximums, 0, 0)
        # replace all label ids, with the maximum value found for that label in its
        # area
        mask_maximums = maximums[labels]

        holes = mask_maximums == 4
        gaps = (mask_maximums < 4) & (mask_maximums > 0)

        self.hole_pixels = np.count_nonzero(holes)
        self.gap_pixels = np.count_nonzero(gaps)

        if self.spatial_qajson:
            self.extents_geojson = ifd.get_extents_feature()

            features = labeled_array_to_geojson(
                labels,
                tile,
                ifd,
                self.pixel_growth
            )

            for feature in features:
                self.tiles_geojson.coordinates.append(
                    feature.geometry.coordinates
                )

        #
        # TODO: add support for 'detailed spatial outputs'
        #

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

        if self.execution_status == "aborted":
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

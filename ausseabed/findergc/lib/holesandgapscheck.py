import geojson
import logging
import numpy as np

from numpy.typing import ArrayLike
from typing import List

from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.gridcheck import GridCheck, GridCheckState
from ausseabed.mbesgc.lib.tiling import Tile


logger = logging.getLogger(__name__)


class HolesAndGapsCheck(GridCheck):
    '''
    Checks for holes and data gaps as indicated by patches of nodata
    '''

    id = 'd095f584-a4aa-4f92-b123-336e1e328289'
    name = 'Hole and Data Gaps Check'
    version = '1'

    # no input params required for this check
    input_params = []

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

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

        # this check only requires the depth layer, so check it is given
        # if not mark this check as aborted
        self.missing_depth = depth is None
        if self.missing_depth:
            self.execution_status = "aborted"
            self.error_message = "Missing depth data"
            logger.info(f"{self.error_message}, aborting hole and data gaps check")
            # we cant run the check so return
            return

        # if there's no pink chart data then use the number of non-nodata cells
        # as the cell count. Otherwise count the number of pinkchart cells
        self.total_cell_count = int(depth.count())
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

        # the mask tells us what pixels are nodata (could be holes)
        # use getmaskarray in place of getmask as getmask will return an
        # empty dimensionless array when there is no nodata in the depth
        # array.
        mask = np.ma.getmaskarray(depth)

        if pinkchart is not None:
            mask = (mask & pinkchart)
        
        #
        # TODO: delete the following and add an actual implementation for this check
        #
        self.gap_count = 0
        self.gap_pixels = 0
        self.hole_count = 2
        self.hole_pixels = 2

        #
        # TODO: later, add support for extracting geojson and 'detailed spatial outputs'
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

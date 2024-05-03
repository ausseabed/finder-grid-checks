import numpy as np
import numpy.ma as ma
import unittest

from scipy.ndimage import find_objects, label

from ausseabed.qajson.model import QajsonParam
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.findergc.lib.holeandgapcheck import HoleAndGapCheck


class TestHoleAndGapCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # create input params that are required by the HoleAndGapCheck
        # class
        cls.input_params = [
            QajsonParam("Ignore edge holes", True),
            QajsonParam("Minimum soundings per node", 5),
            QajsonParam("Gap area threshold (%)", 2),
            QajsonParam("Hole area threshold (%)", 0),
        ]

        # setup a number of test datasets, more details on these
        # (including diagrams) can be found in the related
        # powerpoint presentation

        tc001 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc001 = np.ma.array(
            np.array(tc001, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc002 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 5, 5, 5, 5],
            [5, 5, 4, 5, 5, 5],
            [5, 5, 5, 4, 5, 5],
            [5, 5, 5, 5, 4, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc002 = np.ma.array(
            np.array(tc002, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc003 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc003 = np.ma.array(
            np.array(tc003, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc004 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 5, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc004 = np.ma.array(
            np.array(tc004, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc005 = [
            [5, 5, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 5, 5, 5],
            [5, 4, 4, 4, 5, 5, 5],
            [5, 4, 4, 4, 5, 5, 5],
            [5, 5, 5, 5, 5, 4, 5],
            [5, 5, 5, 5, 4, 5, 5],
            [5, 5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc005 = np.ma.array(
            np.array(tc005, dtype=np.float32),
            mask=np.full((7, 7), False, dtype=bool)
        )

        tc006 = [
            [5, 5, 5, 5, 5, 5, 5],
            [5, 4, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc006 = np.ma.array(
            np.array(tc006, dtype=np.float32),
            mask=np.full((7, 7), False, dtype=bool)
        )

        tc007 = [
            [5, 5, 5, 5, 5, 5, 5],
            [5, 4, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 4, 5],
            [5, 5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc007 = np.ma.array(
            np.array(tc007, dtype=np.float32),
            mask=np.full((7, 7), False, dtype=bool)
        )

        tc008 = [
            [4, 5, 5, 5, 5, 5],
            [4, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc008 = np.ma.array(
            np.array(tc008, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc009 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 4, 5],
            [5, 4, 4, 4, 4, 5],
            [5, 4, 4, 4, 4, 5],
            [5, 4, 4, 4, 4, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc009 = np.ma.array(
            np.array(tc009, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc010 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 4, 4, 5, 5, 5],
            [5, 5, 5, 4, 4, 5],
            [5, 5, 5, 4, 4, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc010 = np.ma.array(
            np.array(tc010, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc011 = [
            [5, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 4, 4],
            [5, 4, 4, 4, 4, 4],
            [5, 4, 4, 4, 4, 4],
            [5, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc011 = np.ma.array(
            np.array(tc011, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

        tc012 = [
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 4, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5],
        ]
        # create a density dataset based on the above array. 
        cls.tc012 = np.ma.array(
            np.array(tc012, dtype=np.float32),
            mask=np.full((6, 6), False, dtype=bool)
        )

    def get_tc_data(self, density: ma.MaskedArray) -> tuple[InputFileDetails, Tile]:
        """ Helper function to create some of the object we need to be able to run
        the check over
        """
        size_x, size_y = density.shape

        tc_ifd = InputFileDetails()
        tc_ifd.size_x = size_x
        tc_ifd.size_y = size_y
        tc_ifd.geotransform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tc_ifd.projection = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')  # noqa
        tc_tile = Tile(0, 0, size_x, size_y)
        return tc_ifd, tc_tile

    def run_tc(self, tc_density: ma.MaskedArray) -> dict:
        """ Helper function to run the test case density array through the
        HoleAndGapCheck class, returning the output dict
        """
        tc_ifd, tc_tile = self.get_tc_data(tc_density)

        check = HoleAndGapCheck(self.input_params)
        check.run(
            ifd=tc_ifd,
            tile=tc_tile,
            depth=None,
            density=tc_density,
            uncertainty=None,
            pinkchart=None
        )
        # it's usually the check executor that assigns execution_status, but we're
        # just calling the run method direct here so need to do it ourselves
        # otherwise we won't get anything from `get_outputs`
        check.execution_status = 'completed'
        return check.get_outputs()

    def test_tc001(self):
        output = self.run_tc(self.tc001)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 9)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc002(self):
        output = self.run_tc(self.tc002)

        # self.assertEqual(output.data["total_hole_count"],0)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 1)
        self.assertEqual(output.data["total_gap_cell_count"], 4)

    def test_tc003(self):
        output = self.run_tc(self.tc003)

        # self.assertEqual(output.data["total_hole_count"],0)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 1)
        self.assertEqual(output.data["total_gap_cell_count"], 8)

    def test_tc004(self):
        output = self.run_tc(self.tc004)

        # self.assertEqual(output.data["total_hole_count"],0)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 1)
        self.assertEqual(output.data["total_gap_cell_count"], 7)

    def test_tc005(self):
        output = self.run_tc(self.tc005)

        # self.assertEqual(output.data["total_hole_count"],1)
        self.assertEqual(output.data["total_hole_cell_count"], 9)
        # self.assertEqual(output.data["total_gap_count"], 1)
        self.assertEqual(output.data["total_gap_cell_count"], 2)

    def test_tc006(self):
        output = self.run_tc(self.tc006)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 17)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc007(self):
        output = self.run_tc(self.tc007)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 11)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc008(self):
        output = self.run_tc(self.tc008)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc009(self):
        output = self.run_tc(self.tc009)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 16)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc010(self):
        output = self.run_tc(self.tc010)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 8)

    def test_tc011(self):
        output = self.run_tc(self.tc011)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 0)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

    def test_tc012(self):
        output = self.run_tc(self.tc012)

        # self.assertEqual(output.data["total_hole_count"], 1)
        self.assertEqual(output.data["total_hole_cell_count"], 10)
        # self.assertEqual(output.data["total_gap_count"], 0)
        self.assertEqual(output.data["total_gap_cell_count"], 0)

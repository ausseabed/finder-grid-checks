import numpy as np
import unittest

from scipy.ndimage import find_objects, label

from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.tiling import Tile
from ausseabed.findergc.lib.holesgridcheck import HolesCheck


class TestHolidays(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up some dummy data
        mask = [
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        mask = np.array(mask)
        mask = (mask == 1)
        depth_data = np.random.uniform(
            low=11.0,
            high=21.0,
            size=(
                len(mask),
                len(mask[0])
            )
        )

        density_data = np.random.uniform(
            low=1.0,
            high=10.0,
            size=(
                len(mask),
                len(mask[0])
            )
        )

        uncertainty_data = np.random.uniform(
            low=0.5,
            high=0.9,
            size=(
                len(mask),
                len(mask[0])
            )
        )

        cls.depth = np.ma.array(
            np.array(depth_data, dtype=np.float32),
            mask=mask
        )
        cls.density = np.ma.array(
            np.array(density_data, dtype=np.float32),
            mask=mask
        )
        cls.uncertainty = np.ma.array(
            np.array(uncertainty_data, dtype=np.float32),
            mask=mask
        )

        cls.dummy_ifd = InputFileDetails()
        cls.dummy_ifd.size_x = len(mask)
        cls.dummy_ifd.size_y = len(mask[0])
        cls.dummy_ifd.geotransform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cls.dummy_ifd.projection = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')  # noqa

        cls.dummy_tile = Tile(0, 0, len(mask), len(mask[0]))

    def test_holescheck_with_edges(self):
        input_params = [
            QajsonParam("Ignore edge holes", False)
        ]

        check = HolesCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=self.depth,
            density=self.density,
            uncertainty=self.uncertainty,
            pinkchart=None
        )

        output = check.get_outputs()

        self.assertEqual(output.data["total_hole_count"], 7)

    def test_holescheck_ignore_edges(self):
        input_params = [
            QajsonParam("Ignore edge holes", True)
        ]

        check = HolesCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=self.depth,
            density=self.density,
            uncertainty=self.uncertainty,
            pinkchart=None
        )

        output = check.get_outputs()

        self.assertEqual(output.data["total_hole_count"], 5)

    def test_holescheck_with_edges_inc_pinkchart(self):
        input_params = [
            QajsonParam("Ignore edge holes", False)
        ]

        mask = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        mask = np.array(mask)
        mask = (mask == 1)
        depth_data = np.random.uniform(low=11.0, high=21.0, size=(len(mask), len(mask[0])))
        depth = np.ma.array(
            np.array(depth_data, dtype=np.float32),
            mask=mask
        )

        pc_data = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0]
        ]
        pinkchart = np.array(pc_data, dtype=np.byte)

        check = HolesCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=depth,
            density=None,
            uncertainty=None,
            pinkchart=pinkchart
        )

        self.assertEqual(check.hole_count, 3)
        self.assertEqual(check.hole_pixels, 3)
        self.assertEqual(check.total_cell_count, 13)

    def test_holes(self):
        # code here was used to develop what is now included in the holesgridcheck.py
        # file. Kept in case there is a need to refine this process in future.
        mask = np.ma.getmask(self.depth)

        # this structure will consider diagonal connectivity
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        labeled_array, num_features = label(mask, structure=s)

        print(labeled_array)

        top = labeled_array[0]
        bottom = labeled_array[labeled_array.shape[0] - 1]
        left = labeled_array[:, 0]
        right = labeled_array[:, labeled_array.shape[1] - 1]

        all_edges = np.concatenate((top, bottom, left, right))
        all_edges_unique = np.unique(all_edges)
        all_edges_unique = all_edges_unique[all_edges_unique != 0]

        print("top")
        print(top)
        print("bottom")
        print(bottom)
        print("left")
        print(left)
        print("right")
        print(right)
        print("all_edges")
        print(all_edges_unique)

        object_slices = find_objects(labeled_array)
        for obj_slice in object_slices:
            patch = labeled_array[obj_slice]
            print("patch b4")
            print(patch)
            patch = np.where(
                np.isin(
                    patch,
                    all_edges_unique,
                    assume_unique=True),
                0,
                patch)
            print("patch aft")
            print(patch)

            labeled_array[obj_slice] = patch

        print(labeled_array)

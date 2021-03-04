import numpy
from scipy import ndimage
import numexpr
from numba import jit
import structlog

_LOG = structlog.get_logger()


def laplacian_operator(lap: numpy.ndarray, flag_grid: numpy.ndarray, threshold: float):
    """
    Laplacian operator check.

    Notes:
        The flag_grid is modified inplace which is fine,
        unless <ndarray>.flags.writeable is False
        Locations could be returned, or we return None and write results into the
        flag_grid.
        The Cython code also logs the location of each pixel flagged by the check

        numexpression is used to reduce the memory footprint from the temp arrays
        that get created. We get a speed benefit too.
    """

    # (sixy6e) wierd logic if threshold is positive eg:
    #     lap = [[-9, -5, -4],
    #            [ 9,  5,  4],
    #            [-1,  0,  1]]
    #     th = 5
    #     would result in the entire array being true.
    #     Is this the intended behaviour???
    # The following line seems to only produce correct results with a negative
    # threshold value.
    # locations = numexpr.evaluate("(lap < threshold) | (lap > -threshold)")

    # The folling quote is taken from the QC Tools documentation
    # |  The Laplacian Operator is a measure of curvature at each node. It is
    # |  equivalent to summing the depth gradients of the four nodes adjacent
    # |  (north, south, east, and west) to each node. If the absolute value of
    # |  the Laplacian Operator is greater than four times the flier search
    # |  height, the node will be flagged.
    # Therefore the implementation was changed to that below
    lap_abs = numpy.abs(lap)
    locations = numexpr.evaluate("lap_abs > threshold")

    flag_grid[locations] = 1  # check number 1

    # log the locations
    # if really desired, this could be done differently,
    # even though the locations are written as GeoPoints later on ...
    # for row, col in zip(*numpy.where(locations)):  # numpy.where is slow but fits the need
    #     _LOG.info("laplacian operator check (#1)", row=row, col=col)


def gaussian_curvature(
    gauss_curve: numpy.ndarray, flag_grid: numpy.ndarray, threshold: float
):
    """
    Gaussian curvature check.

    Notes:
        Similar notes in regards to modification in-place as the laplacian operator
        check.
        The operation could be done in 1 line, but the original code logs the locations
        of the flags.
    """

    locations = gauss_curve > threshold
    flag_grid[locations] = 2  # check number 2

    # for row, col in zip(*numpy.where(locations)):
    #     _LOG.info("gaussian curvature check (#2)", row=row, col=col)


@jit(nopython=True, cache=True)
def adjacent_cells(
    bathy: numpy.ndarray,
    flag_grid: numpy.ndarray,
    threshold: float,
    percent_1: float,
    percent_2: float,
):
    """"""

    rows, cols = bathy.shape

    # the grid is traversed row by row

    for row in range(rows):  # we get the row

        # Historically, we were skipping the first and the last row
        # if (row == 0) or (row == rows - 1):
        #     continue

        for col in range(cols):  # we get the column

            # (sixy6e) why not simply loop over [1, n-1]???
            if (col == 0) or (col == cols - 1):
                continue

            if flag_grid[row, col] != 0:  # avoid existing flagged nodes
                continue

            # for each node in the grid, the depth is retrieved
            depth_node = bathy[row, col]

            # any further calculation is skipped in case of a no-data value
            if numpy.isnan(depth_node):
                continue

            neighbour_count = 0  # initialize the number of neighbors
            diff_pos_count = (
                0  # initialize the number of neighbors with positive depth diff
            )
            diff_neg_count = (
                0  # initialize the number of neighbors with negative depth diff
            )

            # ----------- left node -----------

            if col > 0:  # if we are not on the first column

                # attempt to retrieve depth
                if flag_grid[row, col - 1] != 0:
                    continue

                depth_neighbour = bathy[row, col - 1]

                if numpy.isnan(depth_neighbour) and col > 1:
                    if flag_grid[row, col - 2] != 0:
                        continue

                    depth_neighbour = bathy[row, col - 2]

                if numpy.isnan(depth_neighbour) and col > 2:
                    if flag_grid[row, col - 3] != 0:
                        continue

                    depth_neighbour = bathy[row, col - 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- right node -----------

            if col < cols - 1:  # if we are not on the last column

                # attempt to retrieve depth
                if flag_grid[row, col + 1] != 0:
                    continue

                depth_neighbour = bathy[row, col + 1]

                if numpy.isnan(depth_neighbour) and (col < cols - 2):
                    if flag_grid[row, col + 2] != 0:
                        continue

                    depth_neighbour = bathy[row, col + 2]

                if numpy.isnan(depth_neighbour) and (col < cols - 3):
                    if flag_grid[row, col + 3] != 0:
                        continue

                    depth_neighbour = bathy[row, col + 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- bottom node -----------

            if row > 0:  # if we are not on the first row

                # attempt to retrieve depth
                if flag_grid[row - 1, col] != 0:
                    continue

                depth_neighbour = bathy[row - 1, col]

                if numpy.isnan(depth_neighbour) and row > 1:
                    if flag_grid[row - 2, col] != 0:
                        continue

                    depth_neighbour = bathy[row - 2, col]

                if numpy.isnan(depth_neighbour) and row > 2:
                    if flag_grid[row - 3, col] != 0:
                        continue

                    depth_neighbour = bathy[row - 3, col]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- top node -----------

            if row < rows - 1:  # if we are not on the last row

                # attempt to retrieve depth
                if flag_grid[row + 1, col] != 0:
                    continue

                depth_neighbour = bathy[row + 1, col]

                if numpy.isnan(depth_neighbour) and (row < rows - 2):
                    if flag_grid[row + 2, col] != 0:
                        continue

                    depth_neighbour = bathy[row + 2, col]

                if numpy.isnan(depth_neighbour) and (row < rows - 3):
                    if flag_grid[row + 3, col] != 0:
                        continue

                    depth_neighbour = bathy[row + 3, col]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- bottom-left node -----------

            if (row > 0) and (col > 0):  # if we are not on the first row and col

                # attempt to retrieve depth
                if flag_grid[row - 1, col - 1] != 0:
                    continue

                depth_neighbour = bathy[row - 1, col - 1]

                if numpy.isnan(depth_neighbour) and row > 1 and col > 1:
                    if flag_grid[row - 2, col - 2] != 0:
                        continue

                    depth_neighbour = bathy[row - 2, col - 2]

                # if numpy.isnan(depth_neighbour) and row > 2 and col > 2:
                #     if flag_grid[row - 3, col - 3] != 0:
                #         continue
                #     depth_neighbour = bathy[row - 3, col - 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- top-right node -----------

            if (row < rows - 1) and (
                col < cols - 1
            ):  # if we are not on the last row and col

                # attempt to retrieve depth
                if flag_grid[row + 1, col + 1] != 0:
                    continue

                depth_neighbour = bathy[row + 1, col + 1]

                if numpy.isnan(depth_neighbour) and (row < rows - 2) and (col < cols - 2):

                    if flag_grid[row + 2, col + 2] != 0:
                        continue

                    depth_neighbour = bathy[row + 2, col + 2]

                # if numpy.isnan(depth_neighbour) and (row < rows - 3) and (col < cols - 3):
                #     if flag_grid[row + 3, col + 3] != 0:
                #         continue
                #     depth_neighbour = bathy[row + 3, col + 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- bottom-right node -----------

            if (row > 0) and (
                col < cols - 1
            ):  # if we are not on the first row and last col

                # attempt to retrieve depth
                if flag_grid[row - 1, col + 1] != 0:
                    continue

                depth_neighbour = bathy[row - 1, col + 1]

                if numpy.isnan(depth_neighbour) and row > 1 and (col < cols - 2):
                    if flag_grid[row - 2, col + 2] != 0:
                        continue

                    depth_neighbour = bathy[row - 2, col + 2]

                # if numpy.isnan(depth_neighbour) and row > 2 and col > 2:
                #     if flag_grid[row - 3, col + 3] != 0:
                #         continue
                #     depth_neighbour = bathy[row - 3, col + 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            # ----------- top-left node -----------

            if (row < rows - 1) and (
                col > 0
            ):  # if we are not on the last row and first col

                # attempt to retrieve depth
                if flag_grid[row + 1, col - 1] != 0:
                    continue

                depth_neighbour = bathy[row + 1, col - 1]

                if numpy.isnan(depth_neighbour) and (row < rows - 2) and col > 1:
                    if flag_grid[row + 2, col - 2] != 0:
                        continue

                    depth_neighbour = bathy[row + 2, col - 2]

                # if numpy.isnan(depth_neighbour) and (row < rows - 3) and col > 2:
                #     if flag_grid[row + 3, col - 3] != 0:
                #         continue
                #     depth_neighbour = bathy[row + 3, col - 3]

                # evaluate depth difference
                if not numpy.isnan(depth_neighbour):
                    neighbour_count += 1

                    if depth_node - depth_neighbour > threshold:
                        diff_pos_count += 1

                    if depth_node - depth_neighbour < -threshold:
                        diff_neg_count += 1

            if neighbour_count == 0:
                continue

            # (sixy6e) this section prohibts from simply looping over [1, n-1]
            # calculate the ratio among flagged and total neighbors, then use it to
            # decide if a flier
            if (row == 0) or (col == 0) or (row == (rows - 1)) or (col == (cols - 1)):
                thr = 1.0
            elif neighbour_count <= 4:
                thr = percent_1
            else:
                thr = percent_2

            pos_ratio = diff_pos_count / neighbour_count

            if pos_ratio >= thr:
                flag_grid[row, col] = 3  # check #3

                # _LOG.info(
                #     "adjacency check #3",
                #     row=row,
                #     col=col,
                #     diff_pos_count=diff_pos_count,
                #     neighbour_count=neighbour_count,
                #     pos_ratio=pos_ratio,
                #     thr=thr,
                # )

                continue

            neg_ratio = diff_neg_count / neighbour_count
            if neg_ratio >= thr:
                flag_grid[row, col] = 3  # check #3

                # _LOG.info(
                #     "adjacency check #3",
                #     row=row,
                #     col=col,
                #     diff_neg_count=diff_neg_count,
                #     neighbour_count=neighbour_count,
                #     neg_ratio=neg_ratio,
                #     thr=thr,
                # )

                continue


@jit(nopython=True, cache=True)
def noisy_edges(bathy: numpy.ndarray, flag_grid: numpy.ndarray, dist: int, cf: float):
    """"""

    rows, cols = bathy.shape

    # the grid is traversed row by row

    for row in range(rows):  # we get the row

        # (sixy6e) why not simply loop over [1, n-1]???
        if (row == 0) or (row == rows - 1):
            continue

        for col in range(cols):  # we get the column

            # (sixy6e) why not simply loop over [1, n-1]???
            if (col == 0) or (col == cols - 1):
                continue

            if flag_grid[row, col] != 0:  # avoid existing flagged nodes
                continue

            # for each node in the grid, the depth is retrieved
            depth_node = bathy[row, col]

            # any further calculation is skipped in case of a no-data value
            if numpy.isnan(depth_node):
                continue

            neighbour_count = 0
            neighbour_diff = 0.0
            min_depth = -9999.9
            max_diff = 0.0

            # ----------- left node -----------

            # attempt to retrieve depth
            if flag_grid[row, col - 1] != 0:
                continue

            depth_neighbour = bathy[row, col - 1]

            if numpy.isnan(depth_neighbour) and col > 1:
                if flag_grid[row, col - 2] != 0:
                    continue

                depth_neighbour = bathy[row, col - 2]

            if numpy.isnan(depth_neighbour) and col > 2 and dist > 2:
                if flag_grid[row, col - 3] != 0:
                    continue

                depth_neighbour = bathy[row, col - 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- right node -----------

            # attempt to retrieve depth
            if flag_grid[row, col + 1] != 0:
                continue

            depth_neighbour = bathy[row, col + 1]

            if numpy.isnan(depth_neighbour) and (col < cols - 2):
                if flag_grid[row, col + 2] != 0:
                    continue

                depth_neighbour = bathy[row, col + 2]

            if numpy.isnan(depth_neighbour) and (col < cols - 3) and dist > 2:
                if flag_grid[row, col + 3] != 0:
                    continue

                depth_neighbour = bathy[row, col + 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- bottom node -----------

            # attempt to retrieve depth
            if flag_grid[row - 1, col] != 0:
                continue

            depth_neighbour = bathy[row - 1, col]

            if numpy.isnan(depth_neighbour) and row > 1:
                if flag_grid[row - 2, col] != 0:
                    continue

                depth_neighbour = bathy[row - 2, col]

            if numpy.isnan(depth_neighbour) and row > 2 and dist > 2:
                if flag_grid[row - 3, col] != 0:
                    continue

                depth_neighbour = bathy[row - 3, col]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- top node -----------

            # attempt to retrieve depth
            if flag_grid[row + 1, col] != 0:
                continue

            depth_neighbour = bathy[row + 1, col]

            if numpy.isnan(depth_neighbour) and (row < rows - 2):
                if flag_grid[row + 2, col] != 0:
                    continue

                depth_neighbour = bathy[row + 2, col]

            if numpy.isnan(depth_neighbour) and (row < rows - 3) and dist > 2:
                if flag_grid[row + 3, col] != 0:
                    continue

                depth_neighbour = bathy[row + 3, col]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- bottom-left node -----------

            # attempt to retrieve depth
            if flag_grid[row - 1, col - 1] != 0:
                continue

            depth_neighbour = bathy[row - 1, col - 1]

            if numpy.isnan(depth_neighbour) and row > 1 and col > 1 and dist > 2:
                if flag_grid[row - 2, col - 2] != 0:
                    continue

                depth_neighbour = bathy[row - 2, col - 2]

            # if numpy.isnan(depth_neighbour) and row > 2 and col > 2:
            #     if flag_grid[row - 3, col - 3] != 0:
            #         continue
            #     depth_neighbour = bathy[row - 3, col - 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- top-right node -----------

            # attempt to retrieve depth
            if flag_grid[row + 1, col + 1] != 0:
                continue

            depth_neighbour = bathy[row + 1, col + 1]

            if (
                numpy.isnan(depth_neighbour)
                and (row < rows - 2)
                and (col < cols - 2)
                and dist > 2
            ):
                if flag_grid[row + 2, col + 2] != 0:
                    continue

                depth_neighbour = bathy[row + 2, col + 2]

            # if numpy.isnan(depth_neighbour) and (row < rows - 3) and (col < cols - 3):
            #     if flag_grid[row + 3, col + 3] != 0:
            #         continue
            #     depth_neighbour = bathy[row + 3, col + 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- bottom-right node ------------

            # attempt to retrieve depth
            if flag_grid[row - 1, col + 1] != 0:
                continue

            depth_neighbour = bathy[row - 1, col + 1]

            if numpy.isnan(depth_neighbour) and row > 1 and (col < cols - 2) and dist > 2:
                if flag_grid[row - 2, col + 2] != 0:
                    continue

                depth_neighbour = bathy[row - 2, col + 2]

            # if numpy.isnan(depth_neighbour) and row > 2 and col > 2:
            #     if flag_grid[row - 3, col + 3] != 0:
            #         continue
            #     depth_neighbour = bathy[row - 3, col + 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            # ----------- top-left node-----------

            # attempt to retrieve depth
            if flag_grid[row + 1, col - 1] != 0:
                continue

            depth_neighbour = bathy[row + 1, col - 1]

            if numpy.isnan(depth_neighbour) and (row < rows - 2) and col > 1 and dist > 2:
                if flag_grid[row + 2, col - 2] != 0:
                    continue

                depth_neighbour = bathy[row + 2, col - 2]

            # if numpy.isnan(depth_neighbour) and (row < rows - 3) and col > 2:
            #     if flag_grid[row + 3, col - 3] != 0:
            #         continue
            #     depth_neighbour = bathy[row + 3, col - 3]

            # evaluate depth difference
            if not numpy.isnan(depth_neighbour):
                neighbour_count += 1

                if depth_neighbour > min_depth:
                    min_depth = depth_neighbour

                neighbour_diff = abs(depth_node - depth_neighbour)

                if neighbour_diff > max_diff:
                    max_diff = neighbour_diff

            if neighbour_count == 0:
                continue

            if neighbour_count > 6:
                continue

            if min_depth >= -100.0:
                threshold = (0.25 + (0.013 * -min_depth) ** 2) ** 0.5

            else:
                threshold = (1.0 + (0.023 * -min_depth) ** 2) ** 0.5

            if max_diff > cf * threshold:
                flag_grid[row, col] = 6  # check #6

                # _LOG.info(
                #     "noisy neighbour (check #6)",
                #     row=row,
                #     col=col,
                #     neighbour_count=neighbour_count,
                #     max_diff=max_diff,
                #     min_depth=min_depth,
                #     threshold=threshold,
                # )


@jit(nopython=True, cache=True)
def _small_groups(
    img_labels: numpy.ndarray,
    bathy: numpy.ndarray,
    flag_grid: numpy.ndarray,
    th: float,
    area_limit: float,
    check_slivers: bool,
    check_isolated: bool,
    sizes: numpy.ndarray,
):
    """"""
    rows, cols = img_labels.shape

    for i in range(sizes.shape[0]):

        # check only small groups
        if sizes[i] > area_limit:
            continue

        i += 1
        conn_count = 0
        find = False
        for row in range(4, rows - 4):  # skip bbox boundaries

            for col in range(4, cols - 4):  # skip bbox boundaries

                # skip if the cell does not belong to the current small group
                if img_labels[row, col] != i:
                    continue

                last_row, last_col = row, col

                nb_rs = []
                nb_cs = []

                # check for a valid connection to a grid body
                nb_rs.append(row + 1)  # n1
                nb_rs.append(row - 1)  # n2
                nb_rs.append(row - 1)  # n3
                nb_rs.append(row + 1)  # n4
                nb_cs.append(col + 1)  # n1
                nb_cs.append(col + 1)  # n2
                nb_cs.append(col - 1)  # n3
                nb_cs.append(col - 1)  # n4

                nb_rs.append(row + 2)  # n5
                nb_rs.append(row + 2)  # n6
                nb_rs.append(row + 0)  # n7
                nb_rs.append(row - 2)  # n8
                nb_rs.append(row - 2)  # n9
                nb_rs.append(row - 2)  # n10
                nb_rs.append(row + 0)  # n11
                nb_rs.append(row + 2)  # n12
                nb_cs.append(col + 0)  # n5
                nb_cs.append(col + 2)  # n6
                nb_cs.append(col + 2)  # n7
                nb_cs.append(col + 2)  # n8
                nb_cs.append(col + 0)  # n9
                nb_cs.append(col - 2)  # n10
                nb_cs.append(col - 2)  # n11
                nb_cs.append(col - 2)  # n12

                nb_rs.append(row + 3)  # n13
                nb_rs.append(row + 3)  # n14
                nb_rs.append(row + 0)  # n15
                nb_rs.append(row - 3)  # n16
                nb_rs.append(row - 3)  # n17
                nb_rs.append(row - 3)  # n18
                nb_rs.append(row + 0)  # n19
                nb_rs.append(row + 3)  # n20
                nb_cs.append(col + 0)  # n13
                nb_cs.append(col + 3)  # n14
                nb_cs.append(col + 3)  # n15
                nb_cs.append(col + 3)  # n16
                nb_cs.append(col + 0)  # n17
                nb_cs.append(col - 3)  # n18
                nb_cs.append(col - 3)  # n19
                nb_cs.append(col - 3)  # n20

                nb_rs.append(row + 4)  # n21
                nb_rs.append(row + 4)  # n22
                nb_rs.append(row + 0)  # n23
                nb_rs.append(row - 4)  # n24
                nb_rs.append(row - 4)  # n25
                nb_rs.append(row - 4)  # n26
                nb_rs.append(row + 0)  # n27
                nb_rs.append(row + 4)  # n28
                nb_cs.append(col + 0)  # n21
                nb_cs.append(col + 4)  # n22
                nb_cs.append(col + 4)  # n23
                nb_cs.append(col + 4)  # n24
                nb_cs.append(col + 0)  # n25
                nb_cs.append(col - 4)  # n26
                nb_cs.append(col - 4)  # n27
                nb_cs.append(col - 4)  # n28

                nbs_sz = len(nb_rs)

                for ni in range(nbs_sz):

                    nl = img_labels[nb_rs[ni], nb_cs[ni]]
                    if (nl != 0) and (nl != i) and (sizes[nl - 1] > area_limit):
                        conn_count += 1
                        find = True

                        if (
                            abs(bathy[row, col] - bathy[nb_rs[ni], nb_cs[ni]]) > th
                        ) and check_slivers:
                            flag_grid[row, col] = 4  # check #4
                            _LOG.info("check (#4)", ni=ni + 1, row=row, col=col)
                        break

                if find:
                    break

            if find:
                break

        # it is an isolated group
        if (
            (last_row > 4)
            and (last_row < rows - 4)
            and (last_col > 4)
            and (last_col < cols - 4)
        ):
            if (conn_count == 0) and check_isolated:
                flag_grid[last_row, last_col] = 5  # check #5

                _LOG.info("isolated group (#5)", last_row=last_row, last_col=last_col)


def small_groups(
    grid_bin: numpy.ndarray,
    bathy: numpy.ndarray,
    flag_grid: numpy.ndarray,
    th: float,
    area_limit: float,
    check_slivers: bool,
    check_isolated: bool,
):
    """"""

    rows, cols = grid_bin.shape

    img_labels, n_labels = ndimage.label(grid_bin)
    sizes = ndimage.sum(grid_bin, img_labels, range(1, n_labels + 1))

    _small_groups(
        img_labels, bathy, flag_grid, th, area_limit, check_slivers, check_isolated, sizes
    )

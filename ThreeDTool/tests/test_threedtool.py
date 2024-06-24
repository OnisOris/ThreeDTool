from ..threeDTool import *
import numpy as np


class TestThreedtool:
    def test_equal_lines_true(self):
        line1 = Line(0, 0, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert equal_lines(line1, line2)

    def test_equal_lines_false(self):
        line1 = Line(1, 0, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert not equal_lines(line1, line2)

    def test_coplanar_vectors_list(self):
        vector1 = [19, 19, 1]
        vector2 = [-38, -38, -2]
        assert coplanar_vectors(vector1, vector2)

    def test_coplanar_vectors_ndarray(self):
        vector1 = np.array([19, 19, 1])
        vector2 = np.array([-38, -38, -2])
        assert coplanar_vectors(vector1, vector2)

    def test_check_position_lines_3(self):
        line1 = Line(0, 0, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 3

    def test_check_position_lines_2(self):
        line1 = Line(0, 0, 0, 1, 466, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 2

    def test_check_position_lines_1(self):
        line1 = Line(0, 2, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 1

    def test_check_position_lines_0(self):
        line1 = Line(0, 2, 0, 0, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 0

    def point_from_line_line_intersection(self):
        line1 = Line(0, 2, 0, 0, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 0
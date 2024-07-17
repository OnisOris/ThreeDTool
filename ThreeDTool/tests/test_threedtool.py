from ..threeDTool import *
import numpy as np


class TestThreedtool:

    # codirected ---------------------------
    def test_codirected_x(self):
        vector1 = [1, 0, 0]
        vector2 = [2, 0, 0]
        assert codirected(vector1, vector2)

    def test_codirected_y(self):
        vector1 = [0, 1, 0]
        vector2 = [0, 2, 0]
        assert codirected(vector1, vector2)

    def test_codirected_z(self):
        vector1 = [0, 0, 1]
        vector2 = [0, 0, 2]
        assert codirected(vector1, vector2)

    def test_codirected_xyz(self):
        vector1 = [1, 1, 5]
        vector2 = [2, 2, 10]
        vector3 = [-1.5, 1.4, -15000]
        vector4 = [-3, 2.8, -30000]
        vector5 = [1, 1, -5]
        vector6 = [2, 2, -10]
        vector7 = [1.4, 1.67345, 5.456772]
        vector8 = [1.4 * 256, 1.67345 * 256, 5.456772 * 256]
        var = [collinearity_vectors(vector1, vector2),
               collinearity_vectors(vector3, vector4),
               collinearity_vectors(vector5, vector6),
               collinearity_vectors(vector7, vector8)]
        assert np.all(var)

    # collinearity_vectors ---------------------------
    def test_collinearity_vectors(self):
        vector1 = [1, 1, 5]
        vector2 = [-2, -2, -10]
        vector3 = [-1.5, 1.4, -15000]
        vector4 = [-3, 2.8, -30000]
        vector5 = [-1, 1, -5]
        vector6 = [2, -2, 10]
        vector7 = [1.4, 1.67345, 5.456772]
        vector8 = [1.4 * 256, 1.67345 * 256, 5.456772 * 256]
        var = [collinearity_vectors(vector1, vector2),
               collinearity_vectors(vector3, vector4),
               collinearity_vectors(vector5, vector6),
               collinearity_vectors(vector7, vector8)]
        vector1 = [1, 153, 5]
        vector2 = [-345362, -2, -10]
        vector3 = [-1.5346536, 361.4, -15000]
        vector4 = [-3, 2.8, -30000]
        vector5 = [-13567, 145646, -5364363]
        vector6 = [2363636, -23536, 103536]
        vector7 = [-1.44646, 1.673453536, 365.456772]
        vector8 = [1.4353672, -1.67345 * 256, 53455.456772484]
        var2 = [collinearity_vectors(vector1, vector2),
               collinearity_vectors(vector3, vector4),
               collinearity_vectors(vector5, vector6),
               collinearity_vectors(vector7, vector8)]
        assert np.all(var) and not np.all(var2)
    def test_collinearity_vectors_list(self):
        vector1 = [19, 19, 1]
        vector2 = [-38, -38, -2]
        assert collinearity_vectors(vector1, vector2)

    def test_collinearity_vectors_ndarray(self):
        vector1 = np.array([19, 19, 1])
        vector2 = np.array([-38, -38, -2])
        assert collinearity_vectors(vector1, vector2)

    # equal_lines ---------------------------
    def test_equal_lines_true(self):
        line1 = Line(0, 0, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert equal_lines(line1, line2)

    def test_equal_lines_false(self):
        line1 = Line(1, 0, 0, 1, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert not equal_lines(line1, line2)

    # check_position_lines ---------------------------
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

    # point_from_line_line_intersection ---------------------------
    def test_point_from_line_line_intersection(self):
        line1 = Line(0, 2, 0, 0, 1, 1)
        line2 = Line(100, 100, 100, -10000, -10000, -10000)
        assert check_position_lines(line1, line2) == 0

    # position_analyzer_of_plane_plan ---------------------------
    def test_position_analyzer_of_plane_plan(self):
        plane1 = Plane(0, 0, 1, 1)
        plane2 = Plane(1, 0, 0, 0)
        assert position_analyzer_of_plane_plane(plane1, plane2) == 0

    def test_position_analyzer_of_plane_plane_equal(self):
        plane1 = Plane(1, 1, 1, 1)
        plane2 = Plane(-1, -1, -1, -1)
        assert position_analyzer_of_plane_plane(plane1, plane2) == 1

    def test_position_analyzer_of_plane_plane_2(self):
        plane1 = Plane(2, 2, 2, 1)
        plane2 = Plane(2, 2, 2, -1)
        assert position_analyzer_of_plane_plane(plane1, plane2) == 2

    def test_position_analyzer_of_plane_plane_3(self):
        plane1 = Plane(2, 2, 2, 1)
        plane2 = Plane(2, 2, 2, 5)
        assert position_analyzer_of_plane_plane(plane1, plane2) == 3

    # test_point_from_plane_segment_intersection ---------------------------
    def test_point_from_plane_segment_intersection(self):
        plane = Plane(0, 0, 1, 1)
        segment = Line_segment(point1=[1, 1, -2], point2=[1, 1, 2])
        point = point_from_plane_segment_intersection(segment, plane)
        assert np.allclose(point, [1., 1., -1.], 1e-8)

    # test_point_from_plane_line_intersection ---------------------------
    def test_point_from_plane_line_intersection(self):
        plane = Plane(0, 0, 1, 1)
        line = Line(1, 1, 0, 0, 0, 1)
        point = point_from_plane_line_intersection(line, plane)
        assert np.allclose(point, [1., 1., -1.], 1e-8)

    # point_from_line_segment_intersection ---------------------------
    def test_point_from_line_segment_intersection(self):
        segment = Line_segment(point1=[1, 1, 0], point2=[-1, -1, 0])
        line = Line(1, -1, 0, 1, -1, 0)
        point = point_from_line_segment_intersection(line, segment)
        assert np.allclose(point, [0., 0., 0.], 1e-8)

    def test_point_from_line_segment_intersection_not(self):
        segment = Line_segment(point1=[1, 1, 0], point2=[-1, -1, 0])
        line = Line(1, -1, 0, 1, -1, 0)
        point = point_from_line_segment_intersection(line, segment)
        assert np.allclose(point, [0., 0., 0.], 1e-8)
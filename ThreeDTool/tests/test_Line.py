import numpy as np
from ..line import Line
from ..plane import Plane
from ..threeDTool import check_position_lines, equal_lines

class TestLine:
    def test_Line_create(self):
        point = [3, 3, 3]
        vector = [1, 0, 0]
        line = Line(*point, *vector)
        assert np.allclose(point + vector, line.coeffs())

    def test_line_create_from_points(self):
        line = Line()
        point1 = [0, 0, 0]
        point2 = [1, 1, 1]
        line.line_create_from_points(point1, point2)
        assert line.point_belongs_to_the_line([10, 10, 10]) and line.point_belongs_to_the_line(line.offset_point(2))

    def test_line_from_planes(self):
        plane1 = Plane(0, 0, 1, 0)
        plane2 = Plane(1, 1, 0, 0)
        line = Line()
        line.line_from_planes(plane1, plane2)
        line_check = Line()
        line_check.line_create_from_points([10, -10, 0], [-10, 10, 0])
        assert equal_lines(line, line_check)

    def test_point_belongs_to_the_line(self):
        line = Line(0, 0, 0, 1000, 1000, 1000)
        point = [10, 10, 10]
        assert line.point_belongs_to_the_line(point)

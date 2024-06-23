import numpy as np
from ..line import Line_segment
from ..plane import Plane
from ..threeDTool import check_position_lines, equal_lines


class TestLine:
    def test_line_segment_create(self):
        point1 = [3, 3, 3]
        point2 = [0, 0, 0]
        line_s = Line_segment(point1=point1, point2=point2)
        assert line_s.point_belongs_to_the_line([10, 10, 10])

    def test_segment_create_from_points(self):
        line_s = Line_segment()
        point1 = [0, 0, 0]
        point2 = [1, 1, 1]
        line_s.segment_create_from_points(point1, point2)
        assert (line_s.point_belongs_to_the_line([10, 10, 10])
                and line_s.point_belongs_to_the_line(line_s.offset_point(2)))

    def test_line_segment_create_from_point_vector_length(self):
        point = np.array([1, 1, 1])
        vector = [24, 567, 23]
        length = 100
        from numpy import sqrt
        length_vector = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        norm_dot_length = np.array([vector[0] / length_vector,
                                    vector[1] / length_vector,
                                    vector[2] / length_vector]) * length
        line_s = Line_segment()
        line_s.line_segment_create_from_point_vector_lenght(point, vector, length)

        assert np.allclose(line_s.point1, point) and np.allclose(line_s.point2, point + norm_dot_length)

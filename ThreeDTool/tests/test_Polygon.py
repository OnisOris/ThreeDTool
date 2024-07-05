from ..polygon import Polygon
import numpy as np

class TestPolygon:
    def test_polygons_equal(self):
        pol1 = Polygon(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
        pol2 = Polygon(np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]))
        assert pol1.polygons_equal(pol2)

    def test_polygons_equal2(self):
        pol1 = Polygon(np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]))
        pol2 = Polygon(np.array([[1.5, 0, 0], [1.5, 1, 0], [0.5, 1, 0], [0.5, 0, 0]]))
        assert not pol1.polygons_equal(pol2)

    def test_points_from_polygon_polygon_intersection_and_rot_v(self):
        from ..threeDTool import rot_v
        pol1 = Polygon(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
        points_pol = np.array([[1.5, 0, 0], [1.5, 1, 0], [0.5, 1, 0], [0.5, 0, 0]])
        axis = [1, 1, 1]
        points_pol = rot_v(points_pol, np.pi / 6, axis)
        pol2 = Polygon(points_pol)
        points_test = np.array([[1., 0.73205081, 0.], [0.6830127, 0.5, 0.]])
        assert np.allclose(pol1.points_from_polygon_polygon_intersection(pol2), points_test, 1e-8)
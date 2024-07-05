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

    def test_points_from_polygon_polygon_intersection(self):
        pol1 = Polygon(np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]))
        pol2 = Polygon(np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]))
        points = pol1.points_from_polygon_polygon_intersection(pol2)
        print(points)
        assert points
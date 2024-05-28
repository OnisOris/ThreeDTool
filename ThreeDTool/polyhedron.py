from .threeDTool import *
class Polyhedron:
    def __init__(self, triangles):
        self.triangles = triangles
        self.__barycenter = None
        triangles_vertexes = np.array([0, 0, 0])
        for triangle in triangles:
            triangles_vertexes = np.vstack((triangles_vertexes, triangle.get_vertexes()))
        triangles_vertexes = triangles_vertexes[1:np.shape(triangles_vertexes)[0]]
        self.x_min = triangles_vertexes.min(axis=0)[0]
        self.y_min = triangles_vertexes.min(axis=0)[1]
        self.z_min = triangles_vertexes.min(axis=0)[2]
        self.x_max = triangles_vertexes.max(axis=0)[0]
        self.y_max = triangles_vertexes.max(axis=0)[1]
        self.z_max = triangles_vertexes.max(axis=0)[2]
        self.set_barycenter()

    def set_barycenter(self):
        arr = np.array([0, 0, 0])
        for tr in self.triangles:
            arr = np.vstack([arr, tr.get_mean_vertexes()])
        arr = arr[1:]
        xyz_mean = arr.mean(axis=0)
        self.__barycenter = xyz_mean
    def show(self, ax):
        for tr in self.triangles:
            tr.show(ax)


    @property
    def barycenter(self):
        return self.__barycenter
    @barycenter.setter
    def barycenter(self, barycenter):
        self.__barycenter = barycenter
    def point_analyze(self, point: np.ndarray):
        line = Line()
        # tets_point = np.array(self.__barycenter)
        test_point = np.array(self.__barycenter)
        if point_comparison(point, self.barycenter):
            test_point += 0.001
        line.line_create_from_points(point, test_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self.triangles):
            p = beam_triangle_intersection(line, item)
            if np.shape(p) == (3,):
                arr = np.vstack([arr, p])
        # logger.debug(arr)
        # arr = np.round(arr, 2)
        # logger.debug(arr)
        arr = np.unique(arr[1:np.shape(arr)[0]], axis=0)
        idx = np.array([])
        for i, item in enumerate(arr):
            if point_comparison(item, point):
                idx = np.hstack([idx, i])
        if np.shape(idx)[0] != 0:
            idx = idx.astype("int")
            arr = np.delete(arr, idx, axis=0)
        # logger.debug(arr)
        var = (np.shape(arr)[0]) % 2
        # if np.shape(arr)[0] == 1:
        # logger.debug(var)
        if var != 0:
            return True
        else:
            return False
    def get_min_max(self):
        return np.array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])

    def get_median_point(self):
        return np.median(self.get_min_max(), axis=0)

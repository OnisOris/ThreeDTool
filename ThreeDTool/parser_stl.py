import numpy as np
import matplotlib.pyplot as plt
class Parser_stl:
    #def __init__(self):
    def parse_stl(self, file):
        text = file.read()
        index_space = text.find(" ")
        first_n = text.find("\n")
        name = text[index_space + 1:first_n]
        triangles = text.split("facet normal")[1:]
        triangles_array = np.array([])
        for id, triangle in enumerate(triangles):
            t = triangle.split("\n")
            normals_text = t[0][1:].split(" ")
            normal = np.array([])
            for norm in normals_text:
                normal = np.hstack([normal, float(norm)])
            vertex = t[2:5]
            vertex_array = normal
            for vert in vertex:
                index = vert.find("vertex") + 9
                coordinates = np.array(vert[index:].split(" "))
                coordinates = coordinates.astype(float)
                vertex_array = np.vstack([vertex_array, coordinates])
            vertex_array = np.array([vertex_array])

            if id == 0:
                triangles_array = vertex_array
            else:
                triangles_array = np.vstack([triangles_array, vertex_array])
        return triangles_array, name

    def show(self, triangles):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for i, matrix in enumerate(triangles):

            if i == 0:
                x = np.array(matrix[:, 0])
                y = np.array(matrix[:, 1])
                z = np.array(matrix[:, 2])

                x = np.hstack([x, x[1]])[1:]
                y = np.hstack([y, y[1]])[1:]
                z = np.hstack([z, z[1]])[1:]

            else:
                x = np.hstack([x, np.hstack([np.array(matrix[:, 0]), np.array(matrix[:, 0])[1]])[1:]])
                y = np.hstack([y, np.hstack([np.array(matrix[:, 1]), np.array(matrix[:, 1])[1]])[1:]])
                z = np.hstack([z, np.hstack([np.array(matrix[:, 2]), np.array(matrix[:, 2])[1]])[1:]])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        figure = ax.plot(x, y, z, c='r')
        plt.show()

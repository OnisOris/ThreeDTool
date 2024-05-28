from plane import Plane


class Layer:

    def __init__(self, plane: Plane, shape_array, layer_thickness=0.1):
        '''

        :param plane:
        :param layer_thickness:
        '''
        self.layer_thickness = layer_thickness
        self.first_plane = plane
        self.shape_array = shape_array

    def create_second_plane(self, plane: Plane):
        second_plane = plane
        second_plane.d -= self.layer_thickness
        return second_plane



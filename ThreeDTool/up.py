import numpy as np


class Up(np.ndarray):
    """
    Up is universal part.
    Up объединяет в себе функции отображений для точек, векторов, а также построение трехмерного графика из точек.
    При передаче второго аргумента 'v' или v=True, первая точка в массиве будет считаться за начало исхода векторов в
    массиве.
    Если второй аргумент - 'scatter' или scatter=True, то все точки в массиве будут отображаться в качестве точек на
    трехмерном графике.
    Если второй аргумент - 'plot' или plot=True, то из набора точек в массиве построится график
     """

    def __new__(cls, input_array, *args, **kwargs):
        # Преобразуем входной массив в объект нашего класса
        obj = np.asarray(input_array).view(cls)
        # Дополнительная инициализация, если нужно
        obj.attribute = kwargs.get('attribute', None)
        return obj

    def __init__(self, *args, **kwargs):
        """
        :param v: Отображение в виде вектора
        """
        self.color = "green"
        self.v = False
        self.plot = False
        self.scatter = True
        self.s = 12
        self.length = 1
        self.marker = 'o'
        if self.shape == (2,) or self.shape == (3,) or self.shape == (1, 3) or self.shape == (1, 2):
            pass
        elif self.shape[0] > 1:
            if self.shape[1] == 2 or self.shape[1] == 3:
                pass
            else:
                raise Exception("Wrong shape")
        else:
            raise Exception("Wrong shape")

        if len(args) > 1:
            if type(args[1]) == str:
                if args[1] == 'v':
                    if self.check_shape():
                        self.v = True
                        self.scatter = False
                        self.plot = False
                    else:
                        Exception("shape for v need 1x3 | 2x3 | ")
                elif args[1] == 'plot':
                    self.plot = True
                    self.v = False
                    self.scatter = False
                elif args[1] == 'scatter':
                    self.scatter = True
                    self.v = False
                    self.plot = False
            else:
                raise Exception("argument must be str")
        for kwa, arg in kwargs.items():
            if kwa == 'v':
                if type(arg) == bool:
                    self.v = arg
                    # break
                else:
                    raise Exception("v must be bool")
            elif kwa == 'plot':
                if type(arg) == bool:
                    self.plot = arg
                else:
                    raise Exception("plot must be bool")
            elif kwa == 'scatter':
                if type(arg) == bool:
                    self.scatter = arg
                else:
                    raise Exception("plot must be bool")

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Копируем атрибуты из оригинального объекта, если они существуют
        self.attribute = getattr(obj, 'attribute', None)

    def check_shape(self) -> bool:
        """
        Функция проверяет заданную форму матриц для данного класса. Форма - nx2 или nx3, где n (1...n)
        :return: None
        """
        if self.shape == (2,) or self.shape == (3,) or self.shape == (1, 3) or self.shape == (1, 2):
            return True
        elif self.shape[0] > 0 and (self.shape[1] == 2 or self.shape[1] == 3):
            return True
        else:
            return False

    def set_limits(self, ax):
        rad = 0
        for vector in self[1:]:
            rad_new = rad = np.linalg.norm(vector)
            if rad_new > rad:
                rad = rad_new
        point = self[0]
        if self.v:
            xyz_max = point + rad
            xyz_min = point - rad
        else:
            xyz_max = self.max(axis=0)
            xyz_min = self.min(axis=0)
        # коэффициент сцены
        m = 1
        ax.set_xlim(m * xyz_min[0], m * xyz_max[0])
        ax.set_ylim(m * xyz_min[1], m * xyz_max[1])
        if self.shape[1] == 3 or self.shape == (3,):
            ax.set_zlim(m * xyz_min[2], m * xyz_max[2])
        else:
            ax.set_zlim(m * xyz_min[1], m * xyz_max[1])

    def check_plot_v_scatter(self) -> None:
        """
        Данная функция проверяет, чтобы только один из параметров из plot, v, scatter был равен True, если > 1 будет
        True, то  plot = False, v = False, а scatter = True
        """
        arr = np.array([self.plot, self.v, self.scatter])
        co = np.count_nonzero((arr == True))
        if co > 1:
            self.plot = False
            self.v = False
            self.scatter = True

    def show(self, ax) -> None:
        """
        Функция для отображения точек
        :param ax: Объект matplotlib.axes.Axes
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        self.check_plot_v_scatter()
        self.set_limits(ax)
        if self.v:
            if self.shape == (3,) or self.shape == (1, 3):
                ax.quiver(0, 0, 0, *np.squeeze(self),
                          length=self.length, color='r')
            elif self.shape == (2,) or self.shape == (1, 2):
                ax.quiver(0, 0, *np.squeeze[self[0]],
                          length=self.length, color='r')
            elif self.shape[0] > 1 and self.shape[1] == 3:
                for item in self[1:]:
                    ax.quiver(*self[0], *item,
                              length=self.length, color='r')
            elif self.shape[0] > 1 and self.shape[1] == 2:
                for item in self[1:]:
                    ax.quiver(*self[0], 0, *item, 0,
                              length=self.length, color='r')
        elif self.plot:
            if self.shape == (3,) or self.shape == (1, 3):
                ax.plot(self.T[0], self.T[1], self.T[2], color=self.color, marker=self.marker)
            elif self.shape == (2,) or self.shape == (1, 2):
                ax.plot(self.T[0], self.T[1], color=self.color, marker=self.marker)
            elif self.shape[0] > 1 and self.shape[1] == 3:
                ax.plot(self.T[0], self.T[1], self.T[2], color=self.color, marker=self.marker)
            elif self.shape[0] > 1 and self.shape[1] == 2:
                ax.plot(self.T[0], self.T[1], color=self.color, marker=self.marker)
        elif self.scatter:
            if self.shape == (3,) or self.shape == (1, 3):
                ax.scatter(self.T[0], self.T[1], self.T[2], color=self.color, s=self.s, marker=self.marker)
            elif self.shape == (2,) or self.shape == (1, 2):
                ax.scatter(self.T[0], self.T[1], color=self.color, s=self.s, marker=self.marker)
            elif self.shape[0] > 1 and self.shape[1] == 3:
                ax.scatter(self.T[0], self.T[1], self.T[2], color=self.color, s=self.s, marker=self.marker)
            elif self.shape[0] > 1 and self.shape[1] == 2:
                ax.scatter(self.T[0], self.T[1], color=self.color, s=self.s, marker=self.marker)

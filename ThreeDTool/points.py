import numpy as np


class Points(np.ndarray):
    """
    Класс, хранящий в себе точки вида [x, y, z]
    """

    def __new__(cls, input_array,
                signs: list[str] | np.ndarray[list] = None,
                color: str = 'green',
                s: int = 1,
                marker: str = 'o',
                method: str = 'scatter',
                text: bool = False,
                *args, **kwargs):
        """
        :param input_array: Входной массив
        :type input_array: Any
        :param color: Цвет точек или траектории
        :type color: str
        :param s: int
        :param marker: Маркер точек
        :type marker: str
        :param method: Показ точек точками или траекториями
        :type method: str
        :param text: True - подписать точки
        :type text: bool
        """
        # Преобразуем входной массив в объект нашего класса
        obj = np.asarray(input_array).view(cls)

        # Дополнительная инициализация, если нужно
        obj.attribute = kwargs.get('attribute', None)

        cls.signs = signs
        if signs is not None:
            if np.shape(signs)[0] != np.shape(cls)[0]:
                raise Exception("Size 0 axis of signs and xyz must be the same")

        cls.text = text
        cls.method = method
        cls.color = color
        cls.s = s
        cls.marker = marker

            # Возвращаем объект
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        # Копируем атрибуты из оригинального объекта, если они существуют
        self.attribute = getattr(obj, 'attribute', None)


        # self.signs = signs
        # if signs is not None:
        #     if np.shape(signs)[0] != np.shape(self)[0]:
        #         raise Exception("Size 0 axis of signs and xyz must be the same")
        #
        # self.text = text
        # self.method = method
        # # self.xyz = np.array(xyz)
        # self.color = color
        # self.s = s
        # self.marker = marker

    def show(self, ax) -> None:
        """
        Функция для отображения точек
        :param ax: Объект matplotlib.axes.Axes
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        if self.method == 'plot':
            ax.plot(self.T[0], self.T[1], self.T[2], color=self.color)
        elif self.method == 'scatter':
            ax.scatter(self.T[0], self.T[1], self.T[2], color=self.color, s=self.s, marker=self.marker)
        if self.text:
            if self.signs is not None:
                for i, point in enumerate(self):
                    ax.text(point[0], point[1], point[2], "{self.signs[i]}\n{point[0], point[1], point[2]}",
                            color='blue')
            else:
                for i, point in enumerate(self):
                    ax.text(point[0], point[1], point[2], f"{point[0], point[1], point[2]}", color='blue')


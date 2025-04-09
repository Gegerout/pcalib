class Matrix:
    def __init__(self, matrix):
        """
        Инициализирует матрицу в плотном формате.
        :param matrix: Двумерный список, представляющий матрицу.
        """
        if not matrix:
            raise ValueError("Матрица не может быть пустой")
        self.data = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])

    def __eq__(self, other):
        """
        Сравнивает две матрицы.
        """
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data

    def get_element(self, i, j):
        """
        Возвращает элемент матрицы по индексам (i, j)
        """
        return self.data[i][j]

    def get_trace(self):
        """
        Возвращает след матрицы (только для квадратных матриц).
        """
        if self.n != self.m:
            raise ValueError("Матрица должна быть квадратной для вычисления следа")
        return sum(self.data[i][i] for i in range(self.n))

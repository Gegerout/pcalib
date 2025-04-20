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

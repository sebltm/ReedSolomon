import numpy as np


class GaloisField:

    def __init__(self):

        self.size = 512
        self.lowSize = 256
        self.GF = np.zeros(shape=(self.size, 1), dtype=np.int32)
        self.__logTable = np.zeros(shape=(self.lowSize, 1), dtype=np.int32)

        self.GF[0] = 1

        val = 1

        # Initialise GF[0] manually and start at one since log[0] is undefined and therefore unused
        for position in range(1, self.lowSize - 1):
            val <<= 1

            if val & 0x100:
                val ^= 0x11d

            self.GF[position] = val
            self.__logTable[val] = position

        for position in range((self.lowSize - 1), self.size):
            self.GF[position] = self.GF[position - (self.lowSize - 1)]

    def gfMul(self, x: int, y: int) -> int:
        """
        Galois Field multiplication optimised with log table lookup
        :param x: element x
        :param y: element y
        :return: x * y in Galois Field
        """

        if x == 0 or y == 0:
            return 0

        else:
            value = self.__logTable[x][0]
            value += self.__logTable[y][0]

            value = self.GF[value][0]

            return value

    def gfDiv(self, x: int, y: int) -> int:
        """
        Galois Field division optimised with log table lookup
        :param x: element x
        :param y: element y
        :return: x / y in Galois Field
        """
        if y == 0:
            raise ZeroDivisionError()

        if x == 0:
            return 0
        else:
            value = self.__logTable[x][0] - self.__logTable[y][0]
            value += (self.lowSize - 1)
            value = self.GF[value][0]

            return value

    def gfPow(self, x: int, power: int = 2) -> int:
        """
        Performs the calculation x**power using log table lookup in Galois Field arithmetic
        :param x: element x
        :param power: power
        :return: x**power
        """
        element = self.__logTable[x][0] * power % self.lowSize
        return self.GF[element][0]

    def gfInv(self, x):
        """
        Equivalent to 1/x
        :param x: element
        :return: 1/x
        """
        return self.GF[(self.lowSize - 1) - self.__logTable[x][0]][0]

    def __getitem__(self, item):
        """
        Override accessing GF[] from outside the class
        :param item: index of item
        :return: self.GF[item]
        """
        return self.GF[item][0]

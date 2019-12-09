from GaloisField import GaloisField
import copy
import numpy as np


class Polynomial:
    GF = GaloisField()

    def __init__(self, init: list = None):
        if init is None:
            init = [0]

        self.polynomial = init

    def __len__(self):
        return len(self.polynomial)

    def append(self, object):
        self.polynomial.append(object)
        return self

    def __reversed__(self):
        self.polynomial.reverse()
        return self

    def __getitem__(self, key):
        """
        Override []
        :param key: index to get
        :return: item at position self.polynomial[key] or slice
        """
        if isinstance(key, slice):
            return Polynomial(self.polynomial[key.start:key.stop:key.step])

        return self.polynomial[key]

    def __setitem__(self, key, value):
        """
        Override x[key] = value
        :param key: index
        :param value: value to set
        :return: item at position self.polynomial = value
        """
        self.polynomial[key] = value
        return self

    def __add__(self, other):
        """
        Override + operator to perform polynomial addition
        """
        sum = [0] * max(len(self), len(other))

        sum[len(sum) - len(self):len(sum)] = copy.deepcopy(self.polynomial)

        for n in range(len(other)):
            sum[n + len(sum) - len(other)] ^= other[n]

        return Polynomial(sum)

    def __iadd__(self, other):
        """
        Override += operator for polynomial addition
        """
        self.polynomial = self.__add__(other).polynomial
        return self

    def pop(self, key=-1):
        """
        Remove the element at index
        :param key: the index of the element to remove
        :return: the element removed
        """
        item = self.polynomial.pop(key)
        return item

    def scale(self, x):
        """
        Multiply all the elements of the polynomial by a value x
        :param x: the scalar value
        :return: a new scaled Polynomial
        """

        new_polynomial = [self.GF.gfMul(self.polynomial[i], x) for i in range(len(self.polynomial))]
        return Polynomial(new_polynomial)

    def iscale(self, x):
        """
        Scale itself by a value x
        :param x: the scalar value
        :return: itself
        """
        self.polynomial = self.scale(x).polynomial
        return self

    def eval(self, x):
        """
        Evaluate self given x
        :param x: the value to evaluate for
        :return: the result of the evaluation
        """

        val = self.polynomial[0]
        for position in range(1, len(self.polynomial)):
            val = Polynomial.GF.gfMul(val, x) ^ self.polynomial[position]

        return val

    def __mul__(self, other):
        """
        Override * operator for polynomial multiplication
        """
        num = len(self) + len(other) - 1
        mul = [0] * num

        for posY in range(len(other)):
            for posX in range(len(self)):
                mul[posY + posX] ^= Polynomial.GF.gfMul(self[posX], other[posY])

        return Polynomial(mul)

    def __imul__(self, other):
        """
        Override *= for polynomial multiplication
        """
        self.polynomial = self.__mul__(other).polynomial
        return self

    def __truediv__(self, other):
        """
        Override / for polynomial division
        :return: the quotient, the remainder
        """
        num = copy.deepcopy(self)

        for posX in range(len(self) - len(other) - 1):
            for posY in range(1, len(other)):
                num[posX + posY] ^= Polynomial.GF.gfMul(other[posY], num[posX])

        div = -(len(other) - 1)
        return num[:div], num[div:]

    @staticmethod
    def generator(error_size):
        """
        Create the generator polynomial for a given error size
        :param error_size: the given error size
        :return: the generator polynomial for the specific Galois Field with a given error size
        """

        # Initialise the generator polynomial at 1
        polynomial_value = Polynomial([1])

        # Multiply "error_size" consecutive values in the GF
        for position in range(error_size):
            polynomial_value *= Polynomial([1, Polynomial.GF[position]])

        return polynomial_value

    @staticmethod
    def syndromePolynomial(block, error_size):
        """
        Create the syndrome polynomial
        :param block: the block for which we create the syndrome polynomial
        :param error_size: the number of parity symbols
        :return: the syndrome polynomial
        """
        block_polynomial = Polynomial(block)
        generator_polynomial = Polynomial([0] * error_size)

        for i in range(error_size):
            val = Polynomial.GF[i]
            generator_polynomial[i] = block_polynomial.eval(val)

        return generator_polynomial

    @staticmethod
    def errorLocatorPolynomial(error_positions):
        """
        Compute the error locator polynomial
        :param error_positions: the number of error symbols
        :return: the error locator polynomial
        """
        error_locator = Polynomial([1])

        for i in error_positions:
            error_locator *= (Polynomial([1]) + Polynomial([Polynomial.GF.gfPow(2, i), 0]))

        return error_locator

    @staticmethod
    def errorEvaluatorPolynomial(syndrome_polynomial, error_locator, error_size):
        """
        The error evaluator polynomial
        :param syndrome_polynomial: the syndrome polynomial
        :param error_locator: the error locator polynomial
        :param error_size: the number of parity symbols
        :return: the error evaluator polynomial
        """
        _, remainder = (syndrome_polynomial * error_locator) / Polynomial([1] + [0]*error_size)

        return remainder

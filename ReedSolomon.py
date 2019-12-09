from Polynomial import Polynomial
import copy
import random


class ReedSolomon:

    def __init__(self, error_size):
        self.polyObject = Polynomial()
        self.generator_polynomial = Polynomial.generator(error_size=error_size)
        self.GF = self.polyObject.GF

    def encode(self, message: str, error_size: int) -> list:
        """
        Encode a message with parity bits using Reed-Solomon ECC
        :param message: the original message to transmit
        :param error_size: an integer error size
        :return: the message with parity bits as bytes
        """

        # Create a buffer to hold the message and parity bits
        buffer_size = (len(message) + error_size)
        buffer = [0] * buffer_size

        # Encode our message in the buffer
        for position in range(len(message)):
            char = message[position]
            buffer[position] = ord(char)

        # For each character, multiply the bytes of the character with the appropriate term in the generator polynomial
        # Add the error bit created at the end of the buffer
        for position in range(len(message)):
            char = buffer[position]

            # can't calculate log 0!
            if char:
                for poly_position in range(len(self.generator_polynomial)):
                    buffer[position + poly_position] ^= self.GF.gfMul(self.generator_polynomial[poly_position], char)

        for position in range(len(message)):
            char = message[position]
            buffer[position] = ord(char)

        return buffer

    def forneySyndromes(self, syndrome_polynomial: Polynomial, erasures: list, message: list) -> Polynomial:
        """
        Calculate the Forney syndromes - syndromes of the errors in the message independent of the erasures
        :param syndrome_polynomial: the syndrome polynomial
        :param erasures: the list of erasure positions
        :param message: the original message
        :return: a polynomial representation of the non zero Forney syndromes - all 0 indicates no errors
        """

        # Coefficient positions
        erasures = [len(message) - 1 - p for p in erasures]
        forney_syndromes = copy.deepcopy(syndrome_polynomial)

        for i in range(len(erasures)):
            x = self.GF.gfPow(2, erasures[i])

            for j in range(len(forney_syndromes) - 1):
                y = self.GF.gfMul(forney_syndromes[j], x) ^ forney_syndromes[j + 1]
                forney_syndromes[j] = y
            forney_syndromes.pop()

        return forney_syndromes

    def findErrors(self, forney_syndromes: Polynomial, length_message: int) -> list:
        """
        Berlekamp-Massey + Chien search to find the 0s of the error locator polynomial
        :param forney_syndromes: the polynomial representation of the Forney syndromes
        :param length_message: the length of the message + parity bits
        :return: the error locator polynomial
        """

        error_loc_polynomial = Polynomial([1])
        last_known = Polynomial([1])

        # generate the error locator polynomial
        # - Berklekamp-Massey algorithm
        for i in range(0, len(forney_syndromes)):

            # d = S[k] + C[1]*S[k-1] + C[2]*S[k-2] + ... + C[l]*S[k-L]
            # This is the discrepancy delta
            delta = forney_syndromes[i]
            for j in range(1, len(error_loc_polynomial)):
                delta ^= self.GF.gfMul(error_loc_polynomial[-(j+1)], forney_syndromes[i - j])

            # Calculate the next degree of the polynomial
            last_known.append(0)

            # If delta is not 0, correct for it
            if delta != 0:
                if len(last_known) > len(error_loc_polynomial):
                    new_polynomial = last_known.scale(delta)
                    last_known = error_loc_polynomial.scale(self.GF.gfInv(delta))
                    error_loc_polynomial = new_polynomial

                error_loc_polynomial += last_known.scale(delta)

        error_loc_polynomial = error_loc_polynomial[::-1]

        # Stop if too many errors
        error_count = len(error_loc_polynomial) - 1
        if error_count * 2 > len(forney_syndromes):
            raise ReedSolomonError("Too many errors to correct")

        # Find the zeros of the polynomial using Chien search
        error_list = []
        for i in range(self.GF.lowSize):
            error_z = error_loc_polynomial.eval(self.GF.gfPow(2, i))
            if error_z == 0:
                error_list.append(length_message - i - 1)

        # Sanity checking
        if len(error_list) != error_count:
            raise ReedSolomonError("Too many errors to correct")
        else:
            return error_list

    def correct(self, message, syndrome_polynomial: Polynomial, errors: list) -> Polynomial:
        """
        Using the calculated erasures and errors, recover the original message
        :param message: the transmitted message + parity bits
        :param syndrome_polynomial: the syndrome polynomial
        :param errors: a list of erasures + errors
        :return: the decoded and corrected message
        """

        # Calculate error locator polynomial for both erasures and errors
        coefficient_pos = [len(message) - 1 - p for p in errors]
        error_locator = Polynomial.errorLocatorPolynomial(coefficient_pos)

        # Calculate the error evaluator polynomial
        error_eval = Polynomial.errorEvaluatorPolynomial(syndrome_polynomial[::-1], error_locator, len(error_locator))

        # Calculate the error positions polynomial
        error_positions = []
        for i in range(len(coefficient_pos)):
            x = self.GF.lowSize - coefficient_pos[i]
            error_positions.append(self.GF.gfPow(2, -x))

        # This is the Forney algorithm
        error_magnitudes = Polynomial([0] * len(message))
        for i, error in enumerate(error_positions):

            error_inv = self.GF.gfInv(error)

            # Formal derivative of the error locator polynomial
            error_loc_derivative_tmp = Polynomial([])
            for j in range(len(error_positions)):
                if j != i:
                    error_loc_derivative_tmp.append(1 ^ self.GF.gfMul(error_inv, error_positions[j]))

            # Error locator derivative
            error_loc_derivative = 1
            for coef in error_loc_derivative_tmp:
                error_loc_derivative = self.GF.gfMul(error_loc_derivative, coef)

            # Evaluate the error evaluation polynomial according to the inverse of the error
            y = error_eval.eval(error_inv)

            # Compute the magnitude of error
            magnitude = self.GF.gfDiv(y, error_loc_derivative)
            error_magnitudes[errors[i]] = magnitude

        # Correct the message using the error magnitudes
        message_polynomial = Polynomial(message)
        message_polynomial += error_magnitudes
        return message_polynomial

    def decode(self, message: list, error_size: int) -> str:
        """
        :param message: a message with parity bits which might or might not contain errors
        :param error_size: the number of error symbols
        :return: a decoded message if possible
        """
        buffer = copy.deepcopy(message)

        # First check if there's any erasures
        erasures = []
        for position in range(len(buffer)):
            if buffer[position] < 0:
                buffer[position] = 0
                erasures.append(position)

        # Quit if we have too many erasures
        if len(erasures) > error_size:
            raise ReedSolomonError("Too many erasures")

        # Calculate the syndrome polynomial
        syndrome_polynomial = Polynomial.syndromePolynomial(buffer, error_size)
        if max(syndrome_polynomial) == 0:
            return bytearray(buffer[:-error_size]).decode('utf-8')

        # Calculate the Forney syndromes - removes the erasures from the syndrome polynomial
        forney_syndromes = self.forneySyndromes(syndrome_polynomial, erasures, buffer)

        # Calculate a list of errors in the message using Berlekamp-Massey algorithm
        error_list = self.findErrors(forney_syndromes, len(message))
        if error_list is None:
            raise ReedSolomonError("Could not find errors")

        # Correct the erasures and errors in the message using the Forney algorithm
        decoded_symbols = self.correct(buffer, syndrome_polynomial, (erasures + error_list))
        return bytearray(decoded_symbols[:-error_size]).decode('utf-8')


class ReedSolomonError(Exception):
    def __init__(self, message):
        self.message = message


if __name__ == "__main__":
    # Use the same ReedSolomon() object for encoding and decoding! The error size and generator polynomial have to match
    reed_solomon = ReedSolomon(error_size=16)
    transmission = "Hello there, puny humans"

    print("No errors or erasures")
    encoded_block = reed_solomon.encode(message=transmission, error_size=16)
    print("Encoded message:", encoded_block)
    decoded_block = reed_solomon.decode(encoded_block, error_size=16)
    print("Decoded message:", decoded_block)

    print("\nWith one erasure")
    encoded_block = reed_solomon.encode(message=transmission, error_size=16)
    print("Encoded message:", encoded_block)
    encoded_block[0] = -1
    print("Modified message:", encoded_block)
    decoded_block = reed_solomon.decode(encoded_block, error_size=16)
    print("Decoded message:", decoded_block)

    print("\nWith one error")
    encoded_block = reed_solomon.encode(message=transmission, error_size=16)
    print("Encoded message:", encoded_block)
    encoded_block[8] = 12
    print("Modified message:", encoded_block)
    decoded_block = reed_solomon.decode(encoded_block, error_size=16)
    print("Decoded message:", decoded_block)

    print("\nWith one error and one erasure")
    encoded_block = reed_solomon.encode(message=transmission, error_size=16)
    print("Encoded message:", encoded_block)
    encoded_block[0] = 5
    encoded_block[5] = -1
    print("Modified message:", encoded_block)
    decoded_block = reed_solomon.decode(encoded_block, error_size=16)
    print("Decoded message:", decoded_block)

    print("\nWith 15 (maximum) erasures")
    encoded_block = reed_solomon.encode(message=transmission, error_size=16)
    print("Encoded message:", encoded_block)

    numbers_pos = list(range(0, len(encoded_block) - 16))
    positions = random.sample(numbers_pos, 15)
    for pos in positions:
        mod = random.randrange(-50, -1)
        encoded_block[pos] = mod

    print("Modified message:", encoded_block)
    decoded_block = reed_solomon.decode(encoded_block, error_size=16)
    print("Decoded message:", decoded_block)

import multiprocessing
import qrcode
import random
import string
import numpy as np


IMAGE_SIZE = 21
STRING_LENGTH = 10
CHARACTER_SET = string.ascii_lowercase + string.ascii_uppercase + \
    string.digits


def randomString():
    """
    Return a random string where characters are drawn from a fixed
    set of characters.
    """

    return "".join(random.choice(CHARACTER_SET) for _ in range(STRING_LENGTH))


def qrCodeMatrix(data):
    """
    Encode the given data in a QR code and return it as a numpy array of `0`
    and `1`s.
    """

    qr = qrcode.QRCode(
        version=None,  # automatically determine size
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # 7% error tolerance
        box_size=1,  # single pixels for one box
        border=0,  # no border
    )
    qr.add_data(data)
    qr.make(fit=True)  # fit=True automatically determines the size

    # Reverse black <-> white and convert to numpy array
    return 1 - np.asarray(qr.get_matrix(), dtype=np.float)


def dataToVector(data):
    """
    Returns the first letter of the given string, encoded as a numpy vector
    of length `len(CHARACTER_SET)`.
    """

    letter = data[0]
    nums = map(lambda l: l == letter, CHARACTER_SET)
    return np.asarray(list(nums), dtype=np.float)


def getRandomBatch(size):
    """
    Return a number of QR code matrices and the corresponding labels.
    """

    numProcesses = max(1, multiprocessing.cpu_count() - 2)
    pool = multiprocessing.Pool(numProcesses)

    strings = [randomString() for _ in range(size)]
    X = pool.map(qrCodeMatrix, strings)
    y = pool.map(dataToVector, strings)

    X = np.asarray(X)
    X = X[..., np.newaxis]

    pool.close()

    return X, y

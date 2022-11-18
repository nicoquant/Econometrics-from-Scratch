import numpy as np
from numpy.linalg import eig
import xlwings as xw
from time import time
from sklearn.datasets import make_spd_matrix
from scipy.sparse.linalg import norm
import scipy


def numerator_denominator(base_matrix: np.array, j: int, k: int) -> tuple:
    """

    :param base_matrix: matrix
    :param j: columns indices
    :param k: columns indices
    :return: numerator and denominator for eq 11
    """
    num = np.sum(2 * np.dot(base_matrix[:, j], base_matrix[:, k]))
    den = np.sum(
        np.dot(
            (base_matrix[:, j] + base_matrix[:, k]),
            (base_matrix[:, j] - base_matrix[:, k]),
        )
    )
    return num, den


def define_cos_sin_(num: float, den: float) -> tuple or None:
    """

    :param num: numerator (eq 11)
    :param den: denominator (eq 11)
    :return: Parameters to perform the rotation
    """

    if np.abs(num) > np.abs(den):

        cot2 = np.abs(den) / np.abs(num)
        sin2 = 1 / (np.sqrt(1 + cot2 ** 2))
        cos2 = cot2 * sin2

    elif np.abs(num) <= np.abs(den):

        tan2 = np.abs(num) / np.abs(den)
        cos2 = 1 / (np.sqrt(1 + tan2 ** 2))
        sin2 = tan2 * cos2

    elif num == den:
        return

    cos_ = np.sqrt((1 + cos2) / 2)
    sin_ = sin2 / (2 * cos_)

    return cos_, sin_


def sign_def(num: float, den: float, cos_: float, sin_: float) -> tuple:
    """

    :param num: numerator (eq 11)
    :param den: denominator (eq 11)
    :param cos_: eq 14/19
    :param sin_: eq 15/20
    :return: adjusting the signs of the sin_ value
    """

    if den < 0:
        cos_, sin_ = sin_, cos_

    if num != np.round(0, 4):
        sin_ = np.sign(num) * sin_

    return cos_, sin_


def rotate(base_matrix: np.array, j: int, k: int, cos_: float, sin_: float):
    """
    :param base_matrix: matrix
    :param j: columns index
    :param k: columns index
    :param cos_: eq 14/19
    :param sin_: eq 15/20
    :return: single-plane rotations
    """

    base_matrix[:, j], base_matrix[:, k] = (
        base_matrix[:, j] * cos_ + base_matrix[:, k] * sin_,
        -base_matrix[:, j] * sin_ + base_matrix[:, k] * cos_,
    )
    
    return base_matrix[:, j], base_matrix[:, k]  # A



def run(base_matrix: np.array, j: int = 0, k: int = 1, stop: bool = True) -> np.array:
    """

    :param base_matrix: matrix
    :param j: columns indices
    :param k: columns indices
    :param stop: boolean -> activation parameters
    :return: The convergence matrix
    """

    assert (
        np.linalg.norm(base_matrix - np.transpose(base_matrix), scipy.Inf) < 1.0e-6
    )  # Check if matrix is symmetric
    while stop:

        num, den = numerator_denominator(base_matrix, j, k)
        cos_, sin_ = define_cos_sin_(num, den)
        if cos_ == round(1.0, 5) and sin_ < 1.0e-5 and abs(num) < 1.0e-5 and den > 0:
            # Explication: If cos_ = 1 and sin_ = 0, it means that angles have already been adjusted
            # so adjusting them won't change anything. It will always return the previous columns.
            # Since cosinus is a correlation coefficient, if cos_ = 1, it means that the variables are colinear
            # and thus with no angle.
            stop = False  # Rotation is useless

        else:
            cos_, sin_ = sign_def(num, den, cos_, sin_)
            base_matrix[:, j], base_matrix[:, k] = rotate(base_matrix, j, k, cos_, sin_)

        if k < base_matrix.shape[0] - 1:
            stop = True
            k = k + 1
        elif (k == base_matrix.shape[0] - 1) and (j < base_matrix.shape[0] - 2):
            stop = True
            j, k = j + 1, j + 2
        elif not stop:
            return base_matrix
        else:
            stop = True
            j, k = 0, 1


def eigen_vectors(
    conv_mat: np.array, base_matrix: np.array, E_mat: np.array = None
) -> np.array:
    """

    :param conv_mat: convergence matrix
    :param base_matrix: base matrix
    :param E_mat: n * (n+1) matrix of zeros: will stock the results
    :return: matrix with eigenvectors (vecteur d'inercie maximum) and eigenvalues (inercie projeté)
    """
    eig_val = np.linalg.multi_dot([np.transpose(conv_mat), base_matrix, conv_mat])
    E_mat[:, 0] = np.cbrt(np.diag(eig_val))
    E_mat[:, 1:] = normalize_eigenvectors(conv_mat, E_mat)
    return E_mat


def normalize_eigenvectors(conv_mat: np.array, E_mat: np.array) -> np.array:
    """

    :param conv_mat: convergence matrix
    :param E_mat: matrix with the first columns of eigenvalues
    :return: normalisation of the matrix E_mat
    """
    return np.divide(
        conv_mat, E_mat[:, 0], out=np.zeros_like(conv_mat), where=E_mat[:, 0] != 0
    )


if __name__ == "__main__":

    # A = np.array([[2, 1.4, 0], [1.4, 1, 0], [0, 0, 1]])
    A = make_spd_matrix(
        500, random_state=1
    )  # Generate a random symmetric, positive-definite matrix.
    # change the number n_dim to change the dim of the generated matrix
    start = time()
    B = run(A.copy())

    starting_E_mat = np.zeros((B.shape[0], B.shape[0] + 1))
    E_matrix = eigen_vectors(B, A, starting_E_mat.copy())
    end = time()
    print(end-start)
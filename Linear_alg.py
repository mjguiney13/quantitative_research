import numpy as np
import math
from math import sqrt

def add_vectors(u,v):
    assert len(u) == len(v), "Vectors must have equal lengths"
    result = []
    for i in range(len(u)):
        result.append(u[i] + v[i])
    return result

def scalar_multiply(c, v):
    assert isinstance(c, (int, float)), "Scalar must be a number"
    result = []
    for i in range(len(v)):
        result.append(c * v[i])
    return result

def dot_product(u, v):
    result = 0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result

def vector_length(v):
    return sqrt(dot_product(v, v))

def angle_between(u, v):
    dot = dot_product(u, v)
    length = vector_length(u) * vector_length(v)
    angle_radians = math.acos(dot / length)
    return math.degrees(angle_radians)

def normalize(v):
    result = []
    for i in range(len(v)):
        result.append(v[i] / vector_length(v))
    return result

def linear_combination(coefficients, vectors):
    result = [0] * len(vectors[0])
    for i in range(len(coefficients)):
        scaled = scalar_multiply(coefficients[i], vectors[i])
        result = add_vectors(result, scaled)
    return result
def project_scalar(v, w):
    c = dot_product(v, w) / dot_product(v, v)
    return c

def distance_to_plane(normal, d):
    """Computes shortest distance from origin to plane n"""
    result = abs(d) / vector_length(normal)
    return result

def closest_point_on_plane(normal, d):
    """
    Find closest point on plane n * x = d to origin
    :param normal: list of numbers
    :param d: float
    :return: list: closest point to origin
    """
    scalar = d / dot_product(normal, normal)
    return scalar_multiply(scalar, normal)


def is_in_plane(normal, point, d=0):
    """
    Returns True if point lies on the plane n·v = d.

    Parameters
    ----------
    normal : list of numbers
        Normal vector n
    point : list of numbers
        Point to test
    d : float
        Right hand side of plane equation (default 0)

    Returns
    -------
    bool : True if point is on the plane
    """
    assert len(normal) == len(point), "Vectors must have equal length"
    return abs(dot_product(normal, point) - d) < 1e-10

def portfolio_return(weights, returns):
    assert abs(sum(weights) - 1.0) < 1e-10
    return dot_product(weights, returns)

def matrix_multiply(A, B):
    assert len(A[0]) == len(B)
    result = [[0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_transpose(A):
    result = [[0 for j in range(len(A))] for i in range(len(A[0]))]

    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result
if __name__ == "__main__":
    # Test 1: v dotted with itself gives length squared
    v = [3, 4]
    result = dot_product(v, v)
    print(f"dot_product([3, 4], [3, 4]: {result}")
    assert result == 25, "Expected 25"

    # Test 2: Verify against Numpy
    u = [1, 2, 3]
    v = [4, 5, 6]
    result = dot_product(u, v)
    numpy_result = np.dot(u, v)
    assert result == numpy_result, f"Mismatch: {result} vs {numpy_result}"
    print(f"Verified against numpy: {result}")

    # Test 3: perpendicular vectors
    u = [1, 0]
    v = [0, 1]
    result = dot_product(u, v)
    print(f"Perpendicular vectors dot product: {result}")
    assert result == 0, "Perpendicular vectors must have dot product zero"
    print("Perpendicularity confirmed")

    # vector_length tests
    assert abs(vector_length([3, 4]) - 5.0) < 1e-10
    print(f"vector_length([3,4]): {vector_length([3, 4])}")

    # angle_between tests
    assert abs(angle_between([1,0], [0,1]) - 90.0) < 1e-10
    assert abs(angle_between([1,1], [1,0]) - 45.0) < 1e-10
    print(f"angle_between([1,0],[0,1]): {angle_between([1,0],[0,1])}")
    print(f"angle_between([1,1],[1,0]): {angle_between([1,1],[1,0])}")

    # normalize tests
    n = normalize([3, 4])
    assert abs(vector_length(n) - 1.0) < 1e-10
    print(f"normalize([3,4]): {n}")
    print(f"length of normalized vector: {vector_length(n)}")

    # project_scalar tests
    assert abs(project_scalar([1,1], [1,5]) - 3.0) < 1e-10
    assert abs(project_scalar([2,1], [3,4]) - 2.0) < 1e-10
    assert abs(project_scalar([2,3], [4,1]) - 11/13) < 1e-10
    print(f"project_scalar([1,1],[1,5]): {project_scalar([1,1],[1,5])}")
    print(f"project_scalar([2,1],[3,4]): {project_scalar([2,1],[3,4])}")
    print(f"project_scalar([2,3],[4,1]): {project_scalar([2,3],[4,1])}")

    print("\nAll tests passed.")

    #Distance to Plane tests
    print(distance_to_plane([1, 2, 3], 6))
    print(distance_to_plane([2, 1, -2], 9))
    print(distance_to_plane([2, 1, -2], -9))

    #Linear combination
    coefficients = [0.6, 0.4]
    vectors = [[0.03, 0.01], [0.02, 0.04]]
    print(linear_combination([0.6, 0.4], [[0.03, 0.01], [0.02, 0.04]]))

    #Is in plane
    # is_in_plane tests
    print(is_in_plane([1, 2, 3], [1, 1, 1], d=6))
    print(is_in_plane([1, 2, 3], [0, 0, 0], d=6))
    print(is_in_plane([0, 0, 1], [3, 5, 0], d=0))

    # portfolio_return test
    print(portfolio_return([0.6, 0.4], [0.03, 0.01]))  # 0.022
    print(portfolio_return([0.5, 0.3, 0.2], [0.04, 0.02, 0.01]))

    #Matrix Multiplication
    import numpy as np

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    result = matrix_multiply(A, B)
    numpy_result = np.matmul(A, B).tolist()

    print(f"matrix_multiply: {result}")
    print(f"numpy result: {numpy_result}")
    assert result == numpy_result, "Mismatch with NumPy"
    print("Verified against NumPy")

    # Test non-square
    A2 = [[1, 2, 3], [4, 5, 6]]
    B2 = [[7, 8], [9, 10], [11, 12]]
    result2 = matrix_multiply(A2, B2)
    print(f"Non-square result: {result2}")
    assert result2 == np.matmul(A2, B2).tolist()
    print("Non-square verified")

    # Test identity
    I = [[1, 0], [0, 1]]
    v = [[3], [4]]
    print(f"Identity test: {matrix_multiply(I, v)}")

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    AB_T = matrix_transpose(matrix_multiply(A, B))
    BT_AT = matrix_multiply(matrix_transpose(B), matrix_transpose(A))

    print(f"(AB)ᵀ: {AB_T}")
    print(f"BᵀAᵀ: {BT_AT}")
    assert AB_T == BT_AT
    print("(AB)ᵀ = BᵀAᵀ verified")
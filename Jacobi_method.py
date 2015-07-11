#! /usr/bin/python3.3
import numpy as np
from math import sqrt, copysign


def spherical_matrix_norm(matrix):
    """ Returns sum of square of elements in matrix: for diagonal, not diagonal and total """
    diag_sum = sum([matrix[i, i] ** 2 for i in range(len(matrix))])
    non_diag_sum = sum([matrix[i, j] ** 2 for i in range(len(matrix)) for j in range(len(matrix)) if not i == j])
    total_sum = sum([matrix[i, j] ** 2 for i in range(len(matrix)) for j in range(len(matrix))])
    return (diag_sum, non_diag_sum, total_sum)


def max_non_diag_elem(matrix):
    """ Returns indexes (i,j) of max non-diagonal element """
    matrix = np.array(matrix).tolist()
    max_element_value = max([max(x[:k] + x[k + 1:]) for k, x in enumerate(matrix)])
    indexes = [(subarray.index(max_element_value), i) for i, subarray in enumerate(matrix) if max_element_value in subarray and i != subarray.index(max_element_value)]
    return indexes[0]


def matrix_rotarion(matrix, max_i, max_j):
    """ Indexes of max non-diagonal element given, transforms matrix for annulment of this element"""
    res_matrix = np.identity(len(matrix))
    mu = 2 * matrix[max_i, max_j] / (matrix[max_i, max_i] - matrix[max_j, max_j])
    c = sqrt(0.5 * (1 + 1 / sqrt(1 + mu ** 2)))
    s = sqrt(0.5 * (1 - 1 / sqrt(1 + mu ** 2))) * copysign(1, mu)
    res_matrix[max_i, max_i] = c
    res_matrix[max_j, max_j] = c
    res_matrix[max_i, max_j] = s
    res_matrix[max_j, max_i] = -s
    return res_matrix


def transform_matrix(matrix, eps):
    """ Transformes matrix according to Jacobi method """
    while (matrix[max_non_diag_elem(matrix)[0],max_non_diag_elem(matrix)[1]] > eps):        
        elem_i, elem_j = max_non_diag_elem(matrix)
        rotation_step = matrix_rotarion(matrix, elem_i, elem_j)
        matrix = rotation_step.dot(matrix).dot(rotation_step.transpose())  
        # printed results for report
        # print('spherical matrix norm (diag_sum, non_diag_sum, total_sum)\n ',spherical_matrix_norm(matrix))
        # print('T(i,j) \n', rotation_step,'\n')
        # print('T(i,j)-transposed \n', rotation_step.transpose(),'\n')    
    return matrix


def find_eigenvalues(matrix):
    """ Returns eigenvalues - works for already tranformed matrix """
    return ([matrix[i,i] for i in range(len(matrix))])


def main():
    print("NCM: Assignment #4: Finding eigenvalues - Jacobi eigenvalue algorithm \n")
    eps = 0.00001
    A = np.matrix([[6.29,  0.97,   1.00,   1.10],
                   [0.97,  4.13,   1.30,   0.16],
                   [1.00,  1.30,   5.47,   2.10],
                   [1.1,   0.16,   2.10,   6.07]])
    print(' Start with matrix: \n', A, '\n Precision e = ', eps)	
    tranformed = transform_matrix(A,eps)	
    print ('\n Result: \n',find_eigenvalues(tranformed))


if __name__ == '__main__':
    main()
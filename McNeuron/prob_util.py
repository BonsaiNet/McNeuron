"""Collection of tools for probability distributions"""
import numpy as np

def random_unitary_basis(kappa):
    Ax1 = random_2d_rotation_in_3d('x', kappa)
    Ay1 = random_2d_rotation_in_3d('y', kappa)
    Az1 = random_2d_rotation_in_3d('z', kappa)
    Ax2 = random_2d_rotation_in_3d('x', kappa)
    Ay1 = random_2d_rotation_in_3d('y', kappa)
    Az1 = random_2d_rotation_in_3d('z', kappa)
    A = np.dot(np.dot(Ax1,Ay1),Az1)
    B = np.dot(np.dot(Az1,Ay1),Ax1)
    return np.dot(A,B)

def random_2d_rotation_in_3d(axis, kappa):
    theta = np.random.vonmises(0, kappa, 1)
    A = np.eye(3)
    if axis is 'z':
        A[0,0] = np.cos(theta)
        A[1,0] = np.sin(theta)
        A[0,1] = - np.sin(theta)
        A[1,1] = np.cos(theta)
        return A
    if axis is 'y':
        A[0,0] = np.cos(theta)
        A[2,0] = np.sin(theta)
        A[0,2] = - np.sin(theta)
        A[2,2] = np.cos(theta)
        return A
    if axis is 'x':
        A[1,1] = np.cos(theta)
        A[2,1] = np.sin(theta)
        A[1,2] = - np.sin(theta)
        A[2,2] = np.cos(theta)
        return A

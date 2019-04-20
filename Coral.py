import torch
import math
import numpy as np


def CORAL(source, target):
    d = source.data.shape[len(source.data.shape)-1]
    batch_size = source.data.shape[0]
    dimension_number = len(source.data.shape)

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, dimension_number-2, dimension_number-1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, dimension_number-2, dimension_number-1), xmt)
    # frobenius norm between source and target
    #loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    #loss = torch.norm((xc - xct))**2

    covariance = xc - xct
    feature_map_loss = []

    if dimension_number > 2:
        for i in range(covariance.data.shape[1]):
            matrix_2d = []
            for j in range(covariance.data.shape[0]):
                matrix_2d.append(covariance.data[j][i])
            matrix_2d = torch.stack(matrix_2d)
            matrix_2d = matrix_2d.view(matrix_2d.size(0), -1)
            feature_map_loss.append(math.sqrt(torch.trace(torch.matmul(torch.transpose(matrix_2d, 0, 1), matrix_2d)))**2/(4*d*d*batch_size))
        loss = np.mean(feature_map_loss)
    else:
        loss = math.sqrt(torch.trace(torch.matmul(torch.transpose(covariance, dimension_number-2, dimension_number-1),
                                                  (covariance))))**2/(4*d*d*batch_size)
    return loss

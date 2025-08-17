import pywt
import numpy as np

import torch
import pywt
import numpy as np
import torch
import pywt
import numpy as np

def wavelet_decomposition_and_reconstruction(data, wavelet='db1', mode='symmetric'):

    data_np = data.numpy()
    num_features = data_np.shape[1]
    reconstructed_data_np = np.zeros_like(data_np)

    for feature in range(num_features):
        signal = data_np[:, feature]

        cA, cD = pywt.dwt(signal, wavelet, mode=mode)

        reconstructed_signal = pywt.idwt(cA, cD, wavelet, mode=mode)

        reconstructed_data_np[:, feature] = reconstructed_signal[:data_np.shape[0]]


    reconstructed_data = torch.tensor(reconstructed_data_np, dtype=data.dtype)
    return reconstructed_data





import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_data(file_no, labels_list, psd_data_list, freq, num_splits):
    start_index = (int)(num_splits * file_no)
    end_index = (int)(start_index + num_splits)
    
    labels = np.concatenate(labels_list[start_index:end_index])
    psd_data = np.concatenate(psd_data_list[start_index:end_index])
    freqs = freq

    labels_cat = to_categorical(labels)
    labels_cat = labels_cat[:, 1:]

    psd_train, psd_test, labels_train, labels_test = train_test_split(psd_data, 
                                                                labels_cat, test_size = 0.2, stratify = labels_cat)

    psd_train_amp = psd_train*1e9
    psd_test_amp = psd_test*1e9
    
    return psd_train_amp, psd_test_amp, labels_train, labels_test, freqs

def get_data_bands(psd, freqs):
    delta_r = np.searchsorted(freqs, 4, 'left')
    theta_r = np.searchsorted(freqs, 7, 'left') + 1
    alpha_r = np.searchsorted(freqs, 15, 'left') + 1
    beta_r = np.searchsorted(freqs, 31, 'left') + 1
    
    freqs_delta = freqs[:delta_r]
    freqs_theta = freqs[delta_r:theta_r] 
    freqs_alpha = freqs[theta_r:alpha_r]
    freqs_beta = freqs[alpha_r:beta_r]
    
    psd_delta = psd[:, :, :delta_r]
    psd_theta = psd[:, :, delta_r:theta_r]
    psd_alpha = psd[:, :, theta_r:alpha_r]
    psd_beta = psd[:, :, alpha_r:beta_r]
    
    return psd_delta, freqs_delta, psd_theta, freqs_theta, \
            psd_alpha, freqs_alpha, psd_beta, freqs_beta
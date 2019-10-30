import mne
import numpy as np
import os
import re
import matplotlib.pyplot as pl

from model import get_model
from plot import plot_channel_colours
from utils import get_data, get_data_bands 

DATA_DIR = "./VRCP_files"

# take only the filtered files, name saved as XXX_fil.mff
fil_files = list(filter(re.compile(".*fil.mff").match, os.listdir(DATA_DIR)))

# specific filenames for the project
acym_fname = 'VRCP-CYM-Atharva-044115_fil.mff'
argb_fname = 'VRCP-RGB-Atharva-042629_fil.mff'
hcym_fname = 'VRCP-CYM-HarshaW-120035_fil.mff'
hrgb_fname = 'VRCP-RGB-HarshaW-114603_fil.mff'
scym_fname = 'VRCP-CYM-Shreyash-035410_fil.mff'
srgb_fname = 'VRCP-RGB-Shreyash-034214_fil.mff'

acym_fno = fil_files.index(acym_fname)
argb_fno = fil_files.index(argb_fname)
hcym_fno = fil_files.index(hcym_fname)
hrgb_fno = fil_files.index(hrgb_fname)
scym_fno = fil_files.index(scym_fname)
srgb_fno = fil_files.index(srgb_fname)

data_ = []
psd_data_ = []
freqs_ = []
labels_ = []

time_of_event = 5 # colour was shown for 5 seconds
stride_split = 0.1 # stride of 0.1 seconds while splitting events
time_of_split = 1 # size of one window is 1 second i.e. 1000ms
num_splits = (time_of_event - time_of_split)/stride_split + 1
h_freq = 30

channels = ['E9', 'E10', 'E20']

x = 0
num_ev_fil = np.zeros(len(fil_files), dtype = np.int32)

for ff in fil_files:
    fil_egi = mne.io.read_raw_egi("VRCP/" + ff, verbose = False)
    sfreq = fil_egi.info['sfreq']
    ev = mne.find_events(fil_egi, verbose = False)
    ev = ev[:-1] # dont use the last event as it may not be sampled for 5 seconds, epoch creation will give error
    
    tmin = 0
    tmax = time_of_split - 1/sfreq
    s = 0
    for i in range((int)(num_splits)):
        shifted_ev = mne.event.shift_time_events(ev, [1, 2, 3], i*stride_split, sfreq)
        ep = mne.Epochs(fil_egi, shifted_ev, tmin = tmin, tmax = tmax, picks = channels, baseline = (None, None), verbose = False)
        ep_fil = ep.load_data().filter(l_freq = 0, h_freq = 30)
        psd, freq = mne.time_frequency.psd_multitaper(ep_fil, fmax = 30, verbose = False)
        
        psd_data_.append(psd)
        freqs_.append(np.expand_dims(freq, axis = 1))
        labels_.append(shifted_ev[ep.selection, 2])
        s = s + len(ep.selection)
    num_ev_fil[x] = s
    x = x + 1;

psd_acym_train, psd_acym_test, labels_acym_train, labels_acym_test, freqs_acym = get_data(acym_fno, labels_, psd_data_, freq, num_splits) 
model_acym = get_model()
model_acym.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_acym.fit(psd_acym_train, labels_acym_train, batch_size = 16, epochs = 120, verbose = 2, validation_split = 0.2)
model_acym.evaluate(psd_acym_test, labels_acym_test)

psd_argb_train, psd_argb_test, labels_argb_train, labels_argb_test, freqs_argb = get_data(argb_fno, labels_, psd_data_, freq, num_splits)
model_argb = get_model()
model_argb.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_argb.fit(psd_argb_train, labels_argb_train, batch_size = 16, epochs = 120, verbose = 2, validation_split = 0.2)
model_argb.evaluate(psd_argb_test, labels_argb_test)

for i in range(3):
	plot_channel_colours(psd_acym_train, psd_argb_train, labels_acym_train, labels_argb_train, freqs_acym, freqs_argb, 0)

psd_acym_train_delta, freqs_acym_delta, psd_acym_train_theta, freqs_acym_theta, \
    psd_acym_train_alpha, freqs_acym_alpha, \
    psd_acym_train_beta, freqs_acym_beta = get_data_bands(psd_acym_train, freqs_acym)
psd_argb_train_delta, freqs_argb_delta, psd_argb_train_theta, freqs_argb_theta, \
    psd_argb_train_alpha, freqs_argb_alpha, \
    psd_argb_train_beta, freqs_argb_beta = get_data_bands(psd_argb_train, freqs_argb)


plot_channel_colours(psd_acym_train_alpha, psd_argb_train_alpha, labels_acym_train, labels_argb_train, freqs_acym_alpha, freqs_argb_alpha, 0)

#split into bands
psd_ac_train_alpha = psd_acym_train_alpha[labels_acym_train[:, 0] == 1]
psd_ay_train_alpha = psd_acym_train_alpha[labels_acym_train[:, 1] == 1]
psd_am_train_alpha = psd_acym_train_alpha[labels_acym_train[:, 2] == 1]

psd_ar_train_alpha = psd_argb_train_alpha[labels_argb_train[:, 0] == 1]
psd_ag_train_alpha = psd_argb_train_alpha[labels_argb_train[:, 1] == 1]
psd_ab_train_alpha = psd_argb_train_alpha[labels_argb_train[:, 2] == 1]

# find the mean of each band
psd_ac_train_alpha_mean = np.mean(psd_ac_train_alpha, 0)
psd_ay_train_alpha_mean = np.mean(psd_ay_train_alpha, 0)
psd_am_train_alpha_mean = np.mean(psd_am_train_alpha, 0)

psd_ar_train_alpha_mean = np.mean(psd_ar_train_alpha, 0)
psd_ag_train_alpha_mean = np.mean(psd_ag_train_alpha, 0)
psd_ab_train_alpha_mean = np.mean(psd_ab_train_alpha, 0)

# plot the means
avgs = {0:(psd_ac_train_alpha_mean, freqs_acym_alpha, 'cyan'), 
        1:(psd_ay_train_alpha_mean, freqs_acym_alpha, 'yellow'), 
        2:(psd_am_train_alpha_mean, freqs_acym_alpha, 'magenta'), 
        3:(psd_ar_train_alpha_mean, freqs_argb_alpha, 'red'), 
        4:(psd_ag_train_alpha_mean, freqs_argb_alpha, 'green'), 
        5:(psd_ab_train_alpha_mean, freqs_argb_alpha, 'blue')}

for j in range(3):
	for i in range(6):
	    pl.plot(avgs[i][1], avgs[i][0][j], color = avgs[i][2])
	    pl.show()
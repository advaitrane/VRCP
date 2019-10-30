import matplotlib.pyplot as pl

def plot_channel_colours(psd_cym, psd_rgb, labels_cym, labels_rgb, freqs_cym, freqs_rgb, channel):
    c1 = {0 : 'cyan', 1 : 'yellow', 2 : 'magenta'}
    c2 = {0 : 'red', 1 : 'green', 2 : 'blue'}

    for col in range(3):
        pl.subplot(2, 3, col+1)#, figsize = (15, 15))
        for i, psd in enumerate(psd_cym):
            if(labels_cym[i, col] == 1):
                pl.plot(freqs_cym[:], psd[channel, :], color = c1[col])

    for col in range(3):
        pl.subplot(2, 3, 3+col+1)#, figsize = (15, 15))
        for i, psd in enumerate(psd_rgb):
            if(labels_rgb[i, col] == 1):
                pl.plot(freqs_rgb[:], psd[channel, :], color = c2[col])

    pl.show()
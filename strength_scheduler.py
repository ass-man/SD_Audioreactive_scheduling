#%%
from time import time
import IPython.display as ipd

from email.headerregistry import ContentTransferEncodingHeader
from operator import le
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
# %%
duration = 120

y, sr = librosa.load(r"C:\Users\santt\Videos\nihiloxica.mp3",duration = duration)
y_harm, y_perc = librosa.effects.hpss(y)

hop_length = 512

frame_length = 512
framerate = 15
# %%

fig, ax = plt.subplots(nrows=3, sharex=True)
librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

librosa.display.waveshow(y, sr=sr, ax=ax[1])
ax[1].set(title='Envelope view, stereo')
ax[1].label_outer()

librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
ax[2].set(title='Multiple waveforms')
ax[2].legend()

# %%
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
ax.set( title='Sample view', ylim=[-1, 1])
librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
ax.label_outer()
ax.legend()
ax2.legend()
# %%
ipd.display( ipd.Audio(y_harm, rate=sr))
ipd.display(ipd.Audio(y_perc, rate=sr))


# %%

# joo eli täs kikkaillaan jotenki äänenvoimakkuus? root mean square energy? perkussiosta pihalle

def getRMSE(y_calc,lower, upper,cap ):
    S = librosa.magphase(librosa.stft(y_calc, window=np.ones, center=False))[0]
    rms = librosa.feature.rms(S=S)
    times = librosa.times_like(rms)

    #tää mäppää arvot väliin x ja y
    nomr = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    l_norm = [lower + (upper - lower) * x for x in nomr]
    for i,n in enumerate(l_norm[0]):
        if(n>cap):
            l_norm[0][i] = cap
    fig, ax = plt.subplots()

    ax.plot(times,l_norm[0], linewidth=1.0)# %%

    key_frames = []
    framerate = 24
    #ajat listaan keyframejen kans
    for i,t in enumerate(times):
        cur_keyframe = int(np.floor(framerate * t))
        key_frames.append([cur_keyframe,l_norm[0][i]])


    framecnt = max(key_frames,key=lambda x:x[0])[0]
    grouped_frames = []
    #groupataan keyframeittain keskiarvot
    for i in range(framecnt):
        a = [t for t in key_frames if t[0] == i]
        av = np.average([x[1] for x in a])
        grouped_frames.append([i,av])
    return grouped_frames
# %%


#strength schedule
lower, upper = 0.15, 1
cap = 0.65
out = ""
for f in getRMSE(y_perc,lower, upper,cap):
    x = 1 - f[1]

    out += f"{f[0]}:({x:.2f})," # inverse koska colab
print(out)
with open("c:/temp/bb.txt","w") as f:
    f.write(out)
# %%

#zoom
out = ""
for f in getRMSE(y_perc):
    x = 1 + ( 0.07 * f[1])
    out += f"{f[0]}:({x:.3f})," # inverse koska colab
with open("c:/temp/aa.txt","w") as f:
    f.write(out)
# %%

from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_dir = "/home/tony/PostDoc/Projects/Proprio-Walk/data/outdoor_slopes_2/rollout_22-03-29_15-05-26/"

data = pd.read_csv(os.path.join(data_dir, './proprioception.csv'))
frames = data['frame_counter']

input_data = data[["time_from_start", "fall_prob"]]

input_data.set_index('time_from_start', inplace=True)

input_data.plot(style = '*', label = 'Fall Prob Over time')
yhat = savgol_filter(data['fall_prob'], 55, 3) # window size 51, polynomial order 3

categorical_target = np.zeros((yhat.shape[0],), dtype=np.float32)
data_freq = 80 # samples per sec
lookhead = 2 # sec
for i in range(yhat.shape[0]):
    start = np.minimum(i + int(lookhead * data_freq), yhat.shape[0]-1)
    end = np.minimum(start + int(data_freq),   
                     yhat.shape[0])
    fall_in_next_second = np.any(yhat[start:end] > 0.61)
    if fall_in_next_second:
        categorical_target[i] = 1
        print(frames[i])
    else:
        categorical_target[i] = 0

plt.plot(data['time_from_start'],yhat, label = "Savgol smoothing")
plt.plot(data['time_from_start'],categorical_target, label = "Network Label")


plt.legend()
plt.title("Fall Probability")
plt.xlabel('Time')
plt.show()

latent_fts = [f'latent_{i}' for i in range(8)]
latent_data = data[latent_fts].values
ts = data['time_from_start']

lookhead = 0.5

smoothed_target = np.zeros_like(latent_data)
predictive_latent = np.zeros_like(latent_data)
for i in range(latent_data.shape[1]):
    smoothed_target[:,i] = savgol_filter(latent_data[:,i], 81, 3)
    predictive_latent[:,i] = np.roll(smoothed_target[:,i],
                                   -lookhead*data_freq)
    # when it is not possible to predict, just copy
    predictive_latent[-lookhead*data_freq:,i] = smoothed_target[-lookhead*data_freq:,i]

for i in range(latent_data.shape[1]):
    plt.plot(ts, latent_data[:,i], label = "Raw Data")
    plt.plot(ts, smoothed_target[:,i], label = "Savgol smoothing")
    plt.plot(ts, predictive_latent[:,i], label = f"Predictive {lookhead}s")
    plt.legend()
    plt.title(f"Latent #{i}")
    plt.xlabel('Time')
    plt.show()



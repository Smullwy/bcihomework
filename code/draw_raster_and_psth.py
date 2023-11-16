import numpy as np
import h5py
from scipy.io import savemat
import scipy.io
import matplotlib.pyplot as plt

#import scipy.io as scio
#np.set_printoptions(threshold=1e6)


p = "./data/indy_20160407_02.mat"
data = scipy.io.loadmat(p)
pos = data['xy_pos']
spikes = data['spks_mat']
print(spikes.shape)
trails_begin_end_time = data['trails_begin_end_time']
stimulus_dir = data['stimulus_dir']
max_time_len = np.max(trails_begin_end_time[1,:]-trails_begin_end_time[0,:]+1)
print(max_time_len)
nuero_id = 147
trails_num = trails_begin_end_time.shape[1]
spikes_trail = np.zeros((trails_num,max_time_len))
for i in range(trails_num):
    spikes_trail_i = spikes[trails_begin_end_time[0,i]:trails_begin_end_time[1,i]+1,nuero_id]
    spikes_trail[i,:spikes_trail_i.shape[0]] = spikes_trail_i
#y_data = convert_pos2bin_finger(pos, win_size, end_T)
n_dirs = 8
idx = [15,9,3,7,11,17,23,19]

# #raster
# for i in range(n_dirs):
#     angle1 = 2*np.pi/n_dirs*i
#     angle2 = 2*np.pi/n_dirs*(i+1)
#     pos = np.logical_and((stimulus_dir>=angle1),(stimulus_dir<angle2))
#     pos = pos.reshape(-1)
#     raster = spikes_trail[pos,:]
#     hist = np.mean(raster, axis=0)
#     print(hist.shape)
#     plt.subplot(5,5,idx[i])
#     for raster_id in range(raster.shape[0]):
#         timestamps = np.argwhere(raster[raster_id,:])
#         plt.scatter(timestamps, np.ones_like(timestamps) * raster_id, marker='|', color='black')

#     plt.xlim(0,1000)
#     #plt.ylim(0,0.3)
#     plt.title("%.3f"%np.mean(hist))
#     # break
# plt.show()


#psth
for i in range(n_dirs):
    angle1 = 2*np.pi/n_dirs*i
    angle2 = 2*np.pi/n_dirs*(i+1)
    pos = np.logical_and((stimulus_dir>=angle1),(stimulus_dir<angle2))
    pos = pos.reshape(-1)
    raster = spikes_trail[pos,:]
    hist = np.mean(raster, axis=0)/0.004
    print(hist.shape)
    plt.subplot(5,5,idx[i])
    plt.bar(np.arange(hist.shape[0])*4,hist,width=4)
    plt.xlim(0,1000)
    plt.ylim(0,100)
    plt.xlabel("t(ms)")
    plt.title(f"{i*45}°~{i*45+45}°")
    # plt.title("%.3f"%np.mean(hist))
    # break
plt.show()

plt.clf()
# #raster
for nuero_id in range(147,148):
    trails_num = trails_begin_end_time.shape[1]
    spikes_trail = np.zeros((trails_num,max_time_len))
    spikes_trail = spikes_trail * 0
    for i in range(trails_num):
        spikes_trail_i = spikes[trails_begin_end_time[0,i]:trails_begin_end_time[1,i]+1,nuero_id]
        spikes_trail[i,:spikes_trail_i.shape[0]] = spikes_trail_i

    for i in range(n_dirs):
        angle1 = 2*np.pi/n_dirs*i
        angle2 = 2*np.pi/n_dirs*(i+1)
        pos = np.logical_and((stimulus_dir>=angle1),(stimulus_dir<angle2))
        pos = pos.reshape(-1)
        raster = spikes_trail[pos,:]#[:20]
        raster_list = [np.where(raster[i,:] > 0)[0] * 4 for i in range(raster.shape[0])]
        plt.subplot(5,5,idx[i])
        plt.eventplot(raster_list,linelengths=1)
        plt.xlim(0,1000)
        plt.ylim(0,80)
        hist = np.mean(spikes_trail[pos,:], axis=0)
        mean = np.mean(hist[:250])/0.004

        plt.title(f"{i*45}°~{i*45+45}°")
        plt.xlabel("t(ms)")
        plt.ylabel("trial id")
    plt.show()
    # plt.show()
    # break

    
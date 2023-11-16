import numpy as np
import h5py
from scipy.io import savemat
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=1e6)
def transform(data_path):
#data_path = r'C:\Users\wy\Downloads\indy_20161005_06.mat'
    mat = h5py.File(data_path)

    timestamp_cnt = mat['t'].shape[1]
    start_timestamp = mat['t'][0,0]
    interval = 0.004

    assert(timestamp_cnt == round((mat['t'][0,-1]-mat['t'][0,0])/interval+1))
    #print(mat['spikes'])
    r,c = mat['spikes'].shape
    channel_cnt = 0
    for i in range(c):
        for j in range(r):
            spike_timestamp = mat[mat['spikes'][j,i]]
            if spike_timestamp.shape[0] == 2:
                continue
            channel_cnt += 1
    #print("c_cnt =",channel_cnt)

    min_t = 0
    max_t = 0
    for i in range(c):
        for j in range(r):
            # print(i+1,j+1)
            spike_timestamp = mat[mat['spikes'][j,i]]
            # print(spike_timestamp.shape)
            if spike_timestamp.shape[0] == 2:
                continue
            
            spike_timestamp = np.array(spike_timestamp)
            #print(spike_timestamp.shape)
            spike_timestamp = spike_timestamp[0]
            spike_idx = np.rint((spike_timestamp-start_timestamp)/interval)
            #print(spike_idx)
            min_t = min(min_t,np.amin(spike_idx))
            max_t = max(max_t,np.amax(spike_idx))
    #print(max_t,min_t,max_t-min_t)
    spks_mat = np.zeros([int(max_t+1),channel_cnt])
    #print(spks_mat.shape)
    min_t = 100
    max_t = 0
    ch = 0

    for j in range(r):
        for i in range(c):
            # print(i+1,j+1)
            spike_timestamp = mat[mat['spikes'][j,i]]
            # print(spike_timestamp.shape)
            if spike_timestamp.shape[0] == 2:
                continue
            spike_timestamp = np.array(spike_timestamp)
            #print(spike_timestamp.shape)
            spike_timestamp = spike_timestamp[0]
            spike_idx = np.rint((spike_timestamp-start_timestamp)/interval).astype(int)

            spike_idx = spike_idx[spike_idx>=0]
            if spike_idx.size == 0:
                continue
            # idx = np.argwhere(spike_idx<0)
            # spike_idx1 = np.delete(spike_idx,idx)
            # print(spike_idx1-spike_idx2)
            spks_mat[spike_idx,ch] = 1
            ch += 1
            #print(spike_idx)
            min_t = min(min_t,np.amin(spike_idx))
            max_t = max(max_t,np.amax(spike_idx))
    #print(max_t,min_t,max_t-min_t)

    finger_pos = mat['cursor_pos']
    finger_pos = np.array(finger_pos)
    xy_pos = finger_pos[:,:]
    xy_pos = np.transpose(xy_pos)
    

    target_pos = mat['target_pos']
    is_change = np.diff(target_pos,1,axis=1)
    posx = np.argwhere(is_change[0])
    posy = np.argwhere(is_change[1])
    change_time = np.union1d(posx,posy)+1
    print(change_time)
    begin_time = np.concatenate([np.array([0]),change_time[:]],axis=-1).reshape(1,-1)
    end_time = np.concatenate([change_time[:],np.array([target_pos.shape[1]])],axis=-1).reshape(1,-1)
    all_trails = np.concatenate([begin_time,end_time],axis=0)
    
    trails_which_len_larger_than_40ms = all_trails[:,np.argwhere(all_trails[1,:]-all_trails[0,:]>=10)[:,0]]
    #print(trails_which_len_larger_than_40ms) # 2*trails
    #计算刺激方向
    trails_bengin_end = trails_which_len_larger_than_40ms
    v_xy = target_pos[:,trails_bengin_end[1,1:]]-target_pos[:,trails_bengin_end[1,:-1]]
    begin_target = target_pos[:,trails_bengin_end[1,:-1]]  # 2*T
    end_target = target_pos[:,trails_bengin_end[1,1:]]  # 2*T

    trails_bengin_end = trails_bengin_end[:,1:]
    v_xy = v_xy.T
    stimulus_dir = np.arctan(v_xy[:,1]/v_xy[:,0])
    stimulus_dir[v_xy[:,0] < 0] += np.pi
    stimulus_dir[stimulus_dir < 0] += 2*np.pi
    mdic = {"xy_pos": xy_pos, "spks_mat": spks_mat, "trails_begin_end_time":trails_bengin_end,"stimulus_dir":stimulus_dir,"begin_target":begin_target,"end_target":end_target}
    # xy_pos :204446 * 2, 
    # spks_mat : 205696 * 315
    # trails_begin_end_time :2*389
    # stimulus_dir :389
    # target_pos : 2*204446
    #20 0000
    pics = 900
    this_target_pos = target_pos[:,0]
    print(np.max(target_pos[0,:]))
    print(np.min(target_pos[0,:]))
    print(np.max(target_pos[1,:]))
    print(np.min(target_pos[1,:]))
    i = 0
    last_j = 0
    j = 0
    last_time_less_than1 = False
    while True:
        print(i,j)
        begin_j = j
        while target_pos[0,j] == this_target_pos[0] and target_pos[1,j] == this_target_pos[1]:
            j += 1
            if(j>204444):
                break
        if(j>204444):
            break
        time = j - last_j
        if time > 10:
            plt.subplot(20,20,i+1)
            if last_time_less_than1:
                plt.scatter(target_pos[0,last_j-2],target_pos[1,last_j-2],c='green')
            plt.scatter(target_pos[0,begin_j-1],target_pos[1,begin_j-1],c='red')
            plt.scatter(target_pos[0,begin_j],target_pos[1,begin_j],c='blue')
            plt.scatter(xy_pos[last_j:j,0],xy_pos[last_j:j,1],c='black',s=0.1)
            plt.xticks([])
            plt.yticks([])
            plt.xlim(-115,55)
            plt.ylim(-20,145)
            plt.title(str(time))
            i += 1
            last_time_less_than1 = False
        else:
            last_time_less_than1 = True
        last_j = j
        this_target_pos = target_pos[:,j]
        

        if(j>204444):
            break
    plt.gcf().set_size_inches(25, 25)
    plt.savefig("./guiji/low_dpi.png")
    plt.clf()
ps = [
    "E:/下载/indy_20160407_02.mat",
]
for p in ps:
    transform(p)
import numpy as np
import h5py
from scipy.io import savemat
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def cal_R2(y,pred_y):
    yhat = pred_y                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    sserr = np.sum((y-yhat)**2)
    return 1 - sserr / sstot
#import scipy.io as scio
#np.set_printoptions(threshold=1e6)


p = "./data/indy_20160407_02.mat"
data = scipy.io.loadmat(p)
pos = data['xy_pos']
spikes = data['spks_mat']
begin_target = data['begin_target']
end_target = data['end_target']

trails_begin_end_time = data['trails_begin_end_time']
stimulus_dir = data['stimulus_dir']

max_time_len = np.max(trails_begin_end_time[1,:]-trails_begin_end_time[0,:]+1)
print(max_time_len)
print(pos.shape)
print(trails_begin_end_time.shape[1])
for i in range(trails_begin_end_time.shape[1]):
    print(i)
    beg = trails_begin_end_time[0,i]
    end = trails_begin_end_time[1,i]
    #print(end-beg)
    plt.subplot(20,20,i+1)
    num = end-beg+1
    plt.scatter(pos[beg:end+1,0],pos[beg:end+1,1],c=np.arange(num)/num,s=0.1)
    plt.scatter(begin_target[0,i],begin_target[1,i],c='red')
    plt.scatter(end_target[0,i],end_target[1,i],c='blue')
    #plt.title(str(stimulus_dir[0,i]/np.pi*180)[:3])
    plt.title(str(num))
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-115,55)
    plt.ylim(-20,145)

plt.gcf().set_size_inches(50, 50)
plt.savefig("./guiji/jhfgf.png")
plt.clf()



    
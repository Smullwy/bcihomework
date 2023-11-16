import numpy as np
from scipy.io import savemat
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Neural_Decoding.decoders import KalmanFilterDecoder
def cal_cc(y,y1):
    ccx = np.corrcoef(y1[:,0],y[:,0])[0,1]
    ccy = np.corrcoef(y1[:,1],y[:,1])[0,1]
    return (ccx+ccy)/2
def accumulate(v,pos0,dt):
    """
    v: T*2
    pos0: 1*2
    """
    pos = np.zeros_like(v)
    pos[0,:] = pos0
    for i in range(pos.shape[0]-1):
        pos[i+1,:] = pos[i,:] + v[i,:]*dt
    return pos
def kf_prediction(X_train, Y_train, X_test, Y_test, dt, use=None):
    # use -> list: 
    # T*...
    #preprocess train
    #lag=0
    if use is None:
        use = [0]
    pos_y_train = Y_train
    print(pos_y_train.shape)
    temp=np.diff(pos_y_train,axis=0)/dt 
    vels_binned=np.concatenate((temp,temp[-1:,:]),axis=0) 
    temp=np.diff(vels_binned,axis=0)/dt 
    acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) 
    l = [pos_y_train,vels_binned,acc_binned]
    y_train_kf = None
    if len(use)==1:
        y_train_kf=l[use[0]]
    elif len(use)==2:
        y_train_kf=np.concatenate((l[use[0]],l[use[1]]),axis=1)
    elif len(use)==3:
        y_train_kf=np.concatenate((pos_y_train,vels_binned,acc_binned),axis=1)

    pos_y_test = Y_test
    temp=np.diff(pos_y_test,axis=0)/dt 
    vels_binned=np.concatenate((temp,temp[-1:,:]),axis=0) 
    temp=np.diff(vels_binned,axis=0)/dt 
    acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) 
    y_test_kf = None
    l = [pos_y_test,vels_binned,acc_binned]
    if len(use)==1:
        y_test_kf=l[use[0]]
    elif len(use)==2:
        y_test_kf=np.concatenate((l[use[0]],l[use[1]]),axis=1)
    elif len(use)==3:
        y_test_kf=np.concatenate((pos_y_test,vels_binned,acc_binned),axis=1)

    # num_examples=X_kf.shape[0]
    # #Re-align data to take lag into account
    # if lag<0:
    #     y_kf=y_kf[-lag:,:]
    #     X_kf=X_kf[0:num_examples+lag,:]
    # if lag>0:
    #     y_kf=y_kf[0:num_examples-lag,:]
    #     X_kf=X_kf[lag:num_examples,:]
    
    # Z-score inputs
    X_kf_train_mean=np.nanmean(X_train,axis=0)
    X_kf_train_std=np.nanstd(X_train,axis=0)
    X_train=(X_train-X_kf_train_mean)/X_kf_train_std
    X_test=(X_test-X_kf_train_mean)/X_kf_train_std

    #Zero-center outputs
    y_kf_train_mean=np.nanmean(y_train_kf,axis=0)
    y_train_kf=y_train_kf-y_kf_train_mean
    y_test_kf=y_test_kf-y_kf_train_mean

    model_kf=KalmanFilterDecoder(C=1)
    model_kf.fit(X_train,y_train_kf)
    y_predicted_kf=model_kf.predict(X_test,y_test_kf)
    return y_predicted_kf+y_kf_train_mean,y_test_kf+y_kf_train_mean
def pos2bin(x, win_size):
    bin_num = x.shape[0]//win_size
    channel = x.shape[1]
    input_records = np.zeros((bin_num, channel))
    for win in range(bin_num):
        start_T = win_size * win
        end_T = win_size * (win + 1)
        spike_train = x[start_T:end_T, :]
        inp = np.mean(spike_train, axis=0)
        input_records[win] = inp
    return input_records

def spikes2bin(x, win_size):
    bin_num = x.shape[0]//win_size
    channel = x.shape[1]
    input_records = np.zeros((bin_num, channel))
    for win in range(bin_num):
        start_T = win_size * win
        end_T = win_size * (win + 1)
        spike_train = x[start_T:end_T, :]
        inp = np.sum(spike_train, axis=0)
        input_records[win] = inp
    return input_records


p = "./data/indy_20160407_02.mat"
data = scipy.io.loadmat(p)
xy_pos = data['xy_pos']
spks_mat = data['spks_mat']
xy_pos = xy_pos[:200000,:]
spks_mat = spks_mat[:200000,:]

bin_size = 100 #ms
origin_bin_size = 4 #ms
x_binned = spikes2bin(spks_mat,bin_size//origin_bin_size)
y_binned = pos2bin(xy_pos,bin_size//origin_bin_size)

# print(x_binned.shape)
# print(y_binned.shape)

#前60%作为训练,后30%作为预测
n_train = int(x_binned.shape[0] * 0.6)
n_test = int(x_binned.shape[0] * 0.3)
train_begin = 0
train_end = train_begin + n_train
test_begin = train_end
test_end = test_begin + n_test
x_train,x_test = x_binned[train_begin:train_end,:],x_binned[test_begin:test_end,:]
y_train,y_test = y_binned[train_begin:train_end,:],y_binned[test_begin:test_end,:]

res_p = []
res_v = []
res_a = []

#只使用位置解码
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[0])
ccx = np.corrcoef(y_pred[:,0],y[:,0])[0,1]
ccy = np.corrcoef(y_pred[:,1],y[:,1])[0,1]
# print(ccx,ccy)
# print((ccx+ccy)/2)
res_p.append(cal_cc(y_pred,y))
res_v.append(0)
res_a.append(0)

#只使用速度解码,将速度累加得到位置
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[1])
pos_pred = accumulate(y_pred,y_test[0,:],bin_size/1000)
pos_caled_gt = accumulate(y,y_test[0,:],bin_size/1000)
pos_gt = y_test
# print(cal_cc(pos_caled_gt,pos_gt))
# print(cal_cc(pos_pred,pos_caled_gt))
# print(cal_cc(pos_pred,pos_gt))
# print(cal_cc(y,y_pred))
res_p.append(cal_cc(pos_pred,pos_caled_gt))
res_v.append(cal_cc(y_pred,y))
res_a.append(0)

#只使用加速度解码
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[2])
v0 = (y_test[1,:]-y_test[0,:])/(bin_size/1000)
p0 = y_test[0,:]
v_pred = accumulate(y_pred,v0,bin_size/1000)
v_caled_gt = accumulate(y,v0,bin_size/1000)
pos_pred = accumulate(v_pred,p0,bin_size/1000)
pos_caled_gt = accumulate(v_caled_gt,p0,bin_size/1000)
# print(cal_cc(y_pred,y))
# print(cal_cc(v_pred,v_caled_gt))
# print(cal_cc(pos_pred,pos_caled_gt))
# print(cal_cc(pos_pred,y_test))
# print(cal_cc(pos_caled_gt,y_test))
res_p.append(cal_cc(pos_pred,pos_caled_gt))
res_v.append(cal_cc(v_pred,v_caled_gt))
res_a.append(cal_cc(y_pred,y))

#使用01
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[0,1])
v0 = (y_test[1,:]-y_test[0,:])/(bin_size/1000)
p0 = y_test[0,:]
res_p.append(cal_cc(y_pred[:,0:2],y[:,0:2]))
res_v.append(cal_cc(y_pred[:,2:4],y[:,2:4]))
res_a.append(0)
#使用02
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[0,2])
v0 = (y_test[1,:]-y_test[0,:])/(bin_size/1000)
p0 = y_test[0,:]
res_p.append(cal_cc(y_pred[:,0:2],y[:,0:2]))
res_a.append(cal_cc(y_pred[:,2:4],y[:,2:4]))
res_v.append(0)
#使用12
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[1,2])
v0 = (y_test[1,:]-y_test[0,:])/(bin_size/1000)
p0 = y_test[0,:]
res_v.append(cal_cc(y_pred[:,0:2],y[:,0:2]))
res_a.append(cal_cc(y_pred[:,2:4],y[:,2:4]))
pos_pred = accumulate(y_pred[:,0:2],p0,bin_size/1000)
pos_caled_gt = accumulate(y[:,0:2],p0,bin_size/1000)
res_p.append(cal_cc(pos_pred,pos_caled_gt))
#使用012
y_pred,y = kf_prediction(x_train,y_train,x_test,y_test,dt=bin_size/1000,use=[0,1,2])
v0 = (y_test[1,:]-y_test[0,:])/(bin_size/1000)
p0 = y_test[0,:]
res_p.append(cal_cc(y_pred[:,0:2],y[:,0:2]))
res_v.append(cal_cc(y_pred[:,2:4],y[:,2:4]))
res_a.append(cal_cc(y_pred[:,4:6],y[:,4:6]))

print(res_p)
print(res_v)
print(res_a)

size = len(res_p)
x = np.arange(size)+1

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, res_p,  width=width, label='cc_p')
for i, count in enumerate(res_p):
    if count == 0:
        continue
    if count > 0:
        plt.text(x[i], count + 0.01, str(count)[:4], ha='center', va='bottom')
    else:
        plt.text(x[i], count - 0.03, str(count)[:5], ha='center', va='bottom')
plt.bar(x + width, res_v, width=width, label='cc_v')
for i, count in enumerate(res_v):
    if count == 0:
        continue
    plt.text(x[i] + width, count + 0.01, str(count)[:4], ha='center', va='bottom')
plt.bar(x + 2 * width, res_a, width=width, label='cc_a')
for i, count in enumerate(res_a):
    if count == 0:
        continue
    plt.text(x[i] + 2*width, count + 0.01, str(count)[:4], ha='center', va='bottom')
labels = ["p","v","a","p,v","p,a","v,a","p,v,a"]
plt.xticks(np.arange(x.shape[0])+1,labels[:len(x)])
plt.legend()
plt.show()
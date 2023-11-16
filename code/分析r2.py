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
print(spikes.shape)
trails_begin_end_time = data['trails_begin_end_time']
stimulus_dir = data['stimulus_dir']
max_time_len = np.max(trails_begin_end_time[1,:]-trails_begin_end_time[0,:]+1)
print(max_time_len)

#y_data = convert_pos2bin_finger(pos, win_size, end_T)
n_dirs = 8
idx = [15,9,3,7,11,17,23,19]
all_r2 = []
for nuero_id in range(spikes.shape[1]):
    trails_num = trails_begin_end_time.shape[1]
    spikes_trail = np.zeros((trails_num,max_time_len))
    spikes_trail = spikes_trail * 0
    for i in range(trails_num):
        spikes_trail_i = spikes[trails_begin_end_time[0,i]:trails_begin_end_time[1,i]+1,nuero_id]
        spikes_trail[i,:spikes_trail_i.shape[0]] = spikes_trail_i
    y = []
    y_error = []
    trails_id_in_each_dir = []
    for i in range(n_dirs):
        angle1 = 2*np.pi/n_dirs*i
        angle2 = 2*np.pi/n_dirs*(i+1)
        pos = np.logical_and((stimulus_dir>=angle1),(stimulus_dir<angle2))
        pos = pos.reshape(-1)
        trails_id_in_each_dir.append(pos)
        raster = spikes_trail[pos,:]#[:20]
        # raster_list = [np.where(raster[i,:] > 0)[0] * 4 for i in range(raster.shape[0])]
        hist = np.mean(spikes_trail[pos,:], axis=0)
        mean_fr = np.mean(hist[:150])/0.004

        frs_in_this_dir = (np.mean(spikes_trail[:,:150],axis=1)/0.004)[pos]
        std_dev = np.std(frs_in_this_dir)
        y.append(mean_fr)
        y_error.append(std_dev)

    #用8个点拟合编码模型
    x_data = np.arange(n_dirs)/n_dirs*2*np.pi+2*np.pi/n_dirs/2
    y_data = np.array(y)
    def cos_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(frequency * x + phase) + offset
    try:
        params, params_covariance = curve_fit(cos_func, x_data, y_data, p0=[2, 1, 0, 0],bounds=([-np.inf,0,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf]))
    except:
        print(nuero_id)
        all_r2.append(-1)
        continue
    
    #在所有点上检验编码模型
    fr = np.mean(spikes_trail[:,:150],axis=1)/0.004
    r2 = cal_R2(fr,cos_func(stimulus_dir,*params))

    all_r2.append(r2)
all_r2 = np.array(all_r2)
print(all_r2)
# plt.hist(all_r2,bins=40)
# plt.show()
bins = np.linspace(0, 1, 11)
print(bins)
hist, bins = np.histogram(all_r2, bins=bins)
hist = hist / spikes.shape[1] * 100
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.bar(bin_centers, hist, width=(bins[1] - bins[0])/2, edgecolor='black')
for i, count in enumerate(hist):
    plt.text(bin_centers[i], count + 0.1, str(count)[:4]+"%", ha='center', va='bottom')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
# 添加标签和标题
plt.xlabel('r2')
plt.ylabel('percentage')
#plt.title('bar chart')
plt.xticks(bins)

# 显示图形
plt.show()

plt.clf()

#绘制前100个R2最大的神经元的tuning curve
ind = np.argsort(-all_r2)
print(ind[0],ind[20])

for nuero_id in range(100):
    trails_num = trails_begin_end_time.shape[1]
    spikes_trail = np.zeros((trails_num,max_time_len))
    spikes_trail = spikes_trail * 0
    for i in range(trails_num):
        spikes_trail_i = spikes[trails_begin_end_time[0,i]:trails_begin_end_time[1,i]+1,ind[nuero_id]]
        spikes_trail[i,:spikes_trail_i.shape[0]] = spikes_trail_i
    y = []
    y_error = []
    trails_id_in_each_dir = []
    for i in range(n_dirs):
        angle1 = 2*np.pi/n_dirs*i
        angle2 = 2*np.pi/n_dirs*(i+1)
        pos = np.logical_and((stimulus_dir>=angle1),(stimulus_dir<angle2))
        pos = pos.reshape(-1)
        trails_id_in_each_dir.append(pos)
        raster = spikes_trail[pos,:]#[:20]
        # raster_list = [np.where(raster[i,:] > 0)[0] * 4 for i in range(raster.shape[0])]
        hist = np.mean(spikes_trail[pos,:], axis=0)
        mean_fr = np.mean(hist[:150])/0.004

        frs_in_this_dir = (np.mean(spikes_trail[:,:150],axis=1)/0.004)[pos]
        std_dev = np.std(frs_in_this_dir)
        y.append(mean_fr)
        y_error.append(std_dev)

    #用8个点拟合编码模型
    x_data = np.arange(n_dirs)/n_dirs*2*np.pi+2*np.pi/n_dirs/2
    y_data = np.array(y)
    def cos_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(frequency * x + phase) + offset
    try:
        params, params_covariance = curve_fit(cos_func, x_data, y_data, p0=[2, 1, 0, 0],bounds=([-np.inf,0,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf]))
    except:
        print(nuero_id)
        plt.subplot(10,10,nuero_id+1)
        plt.errorbar(np.arange(8)*(2*np.pi/8)+2*np.pi/8/2, y, yerr=y_error, fmt='o', color='blue', capsize=5, label='Data with Error Bars')
        plt.xlim(0,2*np.pi)
        plt.title('no curve fit')
        plt.xticks([])
        continue
    
    #在所有点上检验编码模型
    fr = np.mean(spikes_trail[:,:150],axis=1)/0.004
    r2 = cal_R2(fr,cos_func(stimulus_dir,*params))

    plt.subplot(10,10,nuero_id+1)
    #plt.scatter(np.arange(8)*(2*np.pi/8)+2*np.pi/8/2,y)
    plt.errorbar(np.arange(8)*(2*np.pi/8)+2*np.pi/8/2, y, yerr=y_error, fmt='o', color='blue', capsize=5, label='Data with Error Bars')
    xx = np.arange(0,2*np.pi,np.pi/100)
    if r2>0.2:
        plt.plot(xx, cos_func(xx, *params), label='Fitted function', color='red')
    else:
        plt.plot(xx, cos_func(xx, *params), label='Fitted function', color='black')
    plt.xlim(0,2*np.pi)
    plt.title('R2:'+str(r2)[:5])
    plt.xticks([])
    # plt.show()
    # break
#plt.show()
plt.gcf().set_size_inches(25, 25)
plt.savefig("./guiji/qian_100_r2_tuning_curve.png")
plt.clf()




    
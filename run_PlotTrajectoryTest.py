from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, load_model
import matplotlib.pyplot as plt
from AnalyticalModel import *



################################################################################################################

# define train and test path
train_data_path = join("data", "MTMR_28002", "real", "uniform", "N5", 'D5', "dual")
test_data_path = join("data", "MTMR_28002", "real", "random", 'N10','D5')

# load Trajectory Test experiment data
test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu', is_scale=False)
test_input_mat = test_dataset.x_data.numpy()
test_ouput_mat = test_dataset.y_data.numpy()

test_output_hat_mat_List = []


# get predict CAD Model output
MTM_CAD_model = MTM_CAD()
test_output_hat_mat_List.append(MTM_CAD_model.predict(test_input_mat))

# get predict MLSE4POL Model output
MTM_MLSE4POL_Model = MTM_MLSE4POL()
test_output_hat_mat_List.append(MTM_MLSE4POL_Model.predict(test_input_mat))



# get predict DNN output
use_net = 'SinNet'
device = 'cpu'
D = 5
load_model_path = join(train_data_path, "result", "model")
DNN_model = get_model('MTM', use_net, D, device=device)
DNN_model, DNN_IScaler, DNN_OScaler = load_model(load_model_path, use_net, DNN_model)
test_output_hat_mat_List.append(predictNP(DNN_model, test_input_mat, DNN_IScaler, DNN_OScaler))

# plot predict error bar figures
mean_list = []
std_list = []
for i in range(len(test_output_hat_mat_List)):
    err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
    mean_list.append(np.mean(err_output_mat, axis=0).tolist())
    std_list.append(np.std(err_output_mat, axis=0).tolist())


#print(err_output_mat)



jnt_index = np.arange(2,7)


fig, ax = plt.subplots()
w = 0.2
space = 0.2
ax.bar(jnt_index-space, mean_list[0], yerr=std_list[0],  width=w,align='center', alpha=0.5, ecolor='black', capsize=10, label='CAD')
ax.bar(jnt_index, mean_list[1], yerr=std_list[1],  width=w,align='center', alpha=0.5, ecolor='black', capsize=10, label='MLSE4POL')
ax.bar(jnt_index+space, mean_list[2], yerr=std_list[2],  width=w,align='center', alpha=0.5, ecolor='black', capsize=10, label='DNN')
ax.set_ylabel(r'$|\tau_{e}|$')
ax.set_xticks(jnt_index)
ax.set_xticklabels(['Joint '+str(i) for i in jnt_index.tolist()])
ax.set_title('Absolute Predicted Torque Error for Trajectory Test')
ax.yaxis.grid(True)
ax.autoscale(tight=True)
ax.legend()

# Save the figure and show
plt.tight_layout()
#plt.savefig('TrajTest_AbsErr.png')
plt.show()









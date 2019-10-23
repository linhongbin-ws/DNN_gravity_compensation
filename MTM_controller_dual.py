import rospy
from sensor_msgs.msg import JointState
from evaluateTool import predict_lagrangian, predict
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model
import time
import sys
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle



MTM_ARM = 'MTMR'
pub_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
sub_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'
train_data_path = join("data", "MTMR_28002",'real', 'uniform', 'D5N5','')
use_net = 'VanillaBPNet'
D = 5
device = 'cpu'

model = get_model('MTM', use_net, D, device=device)

model.load_state_dict(torch.load(join(train_data_path, 'model', use_net+'.pt')))
with open(join(train_data_path, 'model', use_net+'.pkl'), 'r') as fid:
    input_scaler = cPickle.load(fid)
    output_scaler = cPickle.load(fid)
    if use_net == 'Lagrangian_SinNet':
        delta_q = cPickle.load(fid)
        w_vec = cPickle.load(fid)


model = model.to('cpu')
pub = rospy.Publisher(pub_topic, JointState, queue_size=15)
rospy.init_node(MTM_ARM + 'controller', anonymous=True)
rate = rospy.Rate(10)  # 10hz
mtm_arm = dvrk.mtm(MTM_ARM)


def callback(data):
    global use_net
    global D
    global input_scaler
    global output_scaler
    global pub

    start = time.clock()


    pos_list = []
    effort_list = []
    for i in range(6):
        pos_list.append(data.position[i])
        effort_list.append(data.effort[i])
    pos_arr = np.array(pos_list)
    effort_arr = np.array(effort_list)
    if D == 5:
        pos_arr = pos_arr[:-1]
    #print(pos_arr)
    input = torch.from_numpy(pos_arr).to('cpu').float()
    input = input.unsqueeze(0)
    #print(input)
    if use_net == 'Lagrangian_SinNet':
        global delta_q
        global w_vec
        output = predict_lagrangian(model, input, input_scaler, output_scaler, delta_q, w_vec)
    else:
        output = predict(model, input, input_scaler, output_scaler, 'cpu')
    output = output.squeeze(0)
    output_arr = output.numpy()
    #print('predict:', output_arr)
    #print('measure:',effort_arr[1:6])
    #print('error:',output_arr-effort_arr[1:6])

    msg = JointState()
    msg.effort = []
    msg.effort.append(0)
    msg.effort.append(output_arr[0])
    msg.effort.append(output_arr[1])
    msg.effort.append(output_arr[2])
    msg.effort.append(output_arr[3])
    msg.effort.append(output_arr[4])
    msg.effort.append(0)
    pub.publish(msg)
    elapsed = time.clock()
    elapsed = elapsed - start
    #print "Time spent in (function name) is: ", elapsed
if __name__ == '__main__':
    init_pos = np.array([0.0, 0.0, 0.0, 0.0, 3.1415/2.0, 0.0, 0.0])
    mtm_arm.move_joint(init_pos)
    time.sleep(3)
    sub = rospy.Subscriber(sub_topic, JointState, callback)
    while not rospy.is_shutdown():
        pass

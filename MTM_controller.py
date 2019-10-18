import rospy
from sensor_msgs.msg import JointState
from evaluateTool import predict_lagrangian, predict
import dvrk
import numpy as np
from os.path import join
import torch
import _pickle as cPickle

MTM_ARM = 'MTMR'
pub_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
sub_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'
train_data_path = join("data", "MTMR_28002",'real', 'uniform', 'D5N5')
use_net = 'Lagrangian_SinNet'
D = 5

model = torch.load(join(train_data_path, 'model', use_net+'.pt'))
with open(join(train_data_path, 'model', use_net+'.pkl'), 'r') as fid:
    input_scaler = cPickle.load(fid)
    outnput_scaler = cPickle.load(fid)
    if use_net == 'Lagrangian_SinNet':
        delta_q = cPickle.load(fid)
        w_vec = cPickle.load(fid)

def callback(data):
    global use_net
    global D
    global input_scaler
    global output_scaler
    if D==5:
        position = data.position[1:6]

    input = torch.from_numpy(position).to('cpu').float()

    if use_net == 'Lagrangian_SinNet':
        global delta_q
        global w_vec
        output = predict_lagrangian(model, input, input_scaler, output_scaler, delta_q, w_vec)
    else:
        output = predict(model, input, input_scaler, output_scaler, 'cpu')

    print(output)




model = model.to('cpu')
pub = rospy.Publisher(pub_topic, JointState, queue_size=10)
sub = rospy.Subscriber(sub_topic, JointState, callback)
rospy.init_node(MTM_ARM + 'controller', anonymous=True)
rate = rospy.Rate(10)  # 10hz
mtm_arm = dvrk.mtm(MTM_ARM)




if __name__ == '__main__':
    try:
        mtm_arm.move_joint(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            pub.publish(hello_str)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
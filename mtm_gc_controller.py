import rospy
from sensor_msgs.msg import JointState
import torch
import numpy as np

def sub_cb(data):
    pos = data.position

def predict_model(input_arr, model, input_scaler, output_scaler, device):
    input_arr_norm = torch.from_numpy(input_scaler.transform(input_arr)).to(device).float()
    output_arr_norm = model(input_arr_norm)


ARM_NAME = 'MTMR'
pub_topic = '/dvrk/'+ARM_NAME+'/set_effort_joint'
sub_topic = '/dvrk/'+ARM_NAME+'/state_joint_current'
rospy.init_node('mtm_gc_controller', anonymous=True)
pub = rospy.Publisher(pub_topic, JointState, queue_size=3)
sub = rospy.Subscriber(sub_topic, JointState, sub_cb)


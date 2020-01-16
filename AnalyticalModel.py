import numpy as np
from math import cos, sin
class MTM_CAD():
    def __init__(self):
        self.g = 9.81
        cm2_x = -0.38
        cm2_y = 0.00
        cm2_z = 0.00
        m2 = 0.65

        cm3_x = -0.25
        cm3_y = 0.00
        cm3_z = 0.00
        m3 = 0.04

        cm4_x = 0.0
        cm4_y = -0.084
        cm4_z = -0.12
        m4 = 0.14

        cm5_x = 0.0
        cm5_y = 0.036
        cm5_z = -0.065
        m5 = 0.04

        cm6_x = 0.0
        cm6_y = -0.025
        cm6_z = 0.05
        m6 = 0.05

        L2 = 0.2794
        L3 = 0.3645
        L4_z0 = 0.1506

        counter_balance = 0.54
        cable_offset = 0.33
        drift2 = -cable_offset
        E5 = 0.007321
        drift5 = - 0.0065

        param_vec = np.zeros((14,1))
        param_vec[1-1, 0] = L2 * m2 + L2 * m3 + L2 * m4 + L2 * m5 + L2 * m6 + cm2_x * m2
        param_vec[2-1, 0] = cm2_y * m2
        param_vec[3-1, 0] = L3 * m3 + L3 * m4 + L3 * m5 + L3 * m6 + cm3_x * m3
        param_vec[4-1, 0] = cm4_y * m4 + cm3_z * m3 + L4_z0 * m4 + L4_z0 * m5 + L4_z0 * m6
        param_vec[5-1, 0] = cm5_z * m5 + cm6_y * m6
        param_vec[6-1, 0] = cm6_z * m6
        param_vec[7-1, 0] = cm4_x * m4
        param_vec[8-1, 0] = - cm4_z * m4 + cm5_y * m5
        param_vec[9-1, 0] = cm5_x * m5
        param_vec[10-1, 0] = cm6_x * m6
        param_vec[11-1, 0] = counter_balance
        param_vec[12-1, 0] = drift2
        param_vec[13-1, 0] = E5
        param_vec[14-1, 0] = drift5
        self.param_vec = param_vec
    def predict(self, input_mat):
        output_mat = np.zeros((input_mat.shape[0], 5))
        for i in range(input_mat.shape[0]):
            q2 = input_mat[i,0]
            q3 = input_mat[i,1]
            q4 = input_mat[i,2]
            q5 = input_mat[i,3]
            q6 = input_mat[i,4]
            R_mat = self.regressor(q2, q3, q4, q5, q6)
            tor = R_mat.dot(self.param_vec).reshape(7)
            output_mat[i,:] = tor[1:-1]
        return output_mat

    def regressor(self, q2, q3, q4, q5, q6):
        g =  self.g
        R_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [g * sin(q2), g * cos(q2), g * cos(q2) * cos(q3) - g * sin(q2) * sin(q3), - g * cos(q2) * sin(q3) - g * cos(q3) * sin(q2), g * cos(q4) * sin(q2) * sin(q3) * sin(q5) - g * cos(q3) * cos(q5) * sin(q2) - g * cos(q2) * cos(q3) * cos(q4) * sin(q5) - g * cos(q2) * cos(q5) * sin(q3), g * cos(q2) * cos(q3) * sin(q4) * sin(q6) + g * cos(q2) * cos(q6) * sin(q3) * sin(q5) + g * cos(q3) * cos(q6) * sin(q2) * sin(q5) - g * sin(q2) * sin(q3) * sin(q4) * sin(q6) + g * cos(q4) * cos(q5) * cos(q6) * sin(q2) * sin(q3) - g * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6), g * cos(q2) * cos(q3) * cos(q4) - g * cos(q4) * sin(q2) * sin(q3), g * sin(q2) * sin(q3) * sin(q4) - g * cos(q2) * cos(q3) * sin(q4), g * cos(q2) * cos(q3) * cos(q4) * cos(q5) - g * cos(q3) * sin(q2) * sin(q5) - g * cos(q2) * sin(q3) * sin(q5) - g * cos(q4) * cos(q5) * sin(q2) * sin(q3),
                   g * cos(q2) * cos(q3) * cos(q6) * sin(q4) - g * cos(q6) * sin(q2) * sin(q3) * sin(q4) - g * cos(q2) * sin(q3) * sin(q5) * sin(q6) - g * cos(q3) * sin(q2) * sin(q5) * sin(q6) - g * cos(q4) * cos(q5) * sin(q2) * sin(q3) * sin(q6) + g * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q6), 0, 1, 0, 0],
                  [0, 0, g * cos(q2 + q3), -g * sin(q2 + q3), -(g * (2 * cos(q2) * cos(q5) * sin(q3) + 2 * cos(q3) * cos(q5) * sin(q2) + 2 * cos(q2) * cos(q3) * cos(q4) * sin(q5) - 2 * cos(q4) * sin(q2) * sin(q3) * sin(q5))) / 2, (g * (2 * cos(q2) * cos(q3) * sin(q4) * sin(q6) + 2 * cos(q2) * cos(q6) * sin(q3) * sin(q5) + 2 * cos(q3) * cos(q6) * sin(q2) * sin(q5) - 2 * sin(q2) * sin(q3) * sin(q4) * sin(q6) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6) + 2 * cos(q4) * cos(q5) * cos(q6) * sin(q2) * sin(q3))) / 2, (g * (cos(q2 + q3 + q4) + cos(q2 + q3 - q4))) / 2, (g * (2 * sin(q2) * sin(q3) * sin(q4) - 2 * cos(q2) * cos(q3) * sin(q4))) / 2, -(g * (2 * cos(q2) * sin(q3) * sin(q5) + 2 * cos(q3) * sin(q2) * sin(q5) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) + 2 * cos(q4) * cos(q5) * sin(q2) * sin(q3))) / 2,
                   -(g * (2 * cos(q6) * sin(q2) * sin(q3) * sin(q4) - 2 * cos(q2) * cos(q3) * cos(q6) * sin(q4) + 2 * cos(q2) * sin(q3) * sin(q5) * sin(q6) + 2 * cos(q3) * sin(q2) * sin(q5) * sin(q6) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q6) + 2 * cos(q4) * cos(q5) * sin(q2) * sin(q3) * sin(q6))) / 2, -cos(q2 + q3), 0, 0, 0],
                  [0, 0, 0, 0, g * sin(q2 + q3) * sin(q4) * sin(q5), g * sin(q2 + q3) * (cos(q4) * sin(q6) + cos(q5) * cos(q6) * sin(q4)), -g * sin(q2 + q3) * sin(q4), -g * sin(q2 + q3) * cos(q4), -g * sin(q2 + q3) * cos(q5) * sin(q4), g * sin(q2 + q3) * (cos(q4) * cos(q6) - cos(q5) * sin(q4) * sin(q6)), 0, 0, 0, 0],
                  [0, 0, 0, 0, -g * (cos(q2) * cos(q3) * sin(q5) - sin(q2) * sin(q3) * sin(q5) + cos(q2) * cos(q4) * cos(q5) * sin(q3) + cos(q3) * cos(q4) * cos(q5) * sin(q2)), g * (cos(q5) * cos(q6) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * cos(q5) * cos(q6) + cos(q2) * cos(q4) * cos(q6) * sin(q3) * sin(q5) + cos(q3) * cos(q4) * cos(q6) * sin(q2) * sin(q5)), 0, 0, -g * (cos(q5) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * cos(q5) + cos(q2) * cos(q4) * sin(q3) * sin(q5) + cos(q3) * cos(q4) * sin(q2) * sin(q5)), -g * (cos(q5) * sin(q2) * sin(q3) * sin(q6) - cos(q2) * cos(q3) * cos(q5) * sin(q6) + cos(q2) * cos(q4) * sin(q3) * sin(q5) * sin(q6) + cos(q3) * cos(q4) * sin(q2) * sin(q5) * sin(q6)), 0, 0, q5, 1],
                  [0, 0, 0, 0, 0, g * (cos(q2) * cos(q6) * sin(q3) * sin(q4) + cos(q3) * cos(q6) * sin(q2) * sin(q4) + cos(q2) * cos(q3) * sin(q5) * sin(q6) - sin(q2) * sin(q3) * sin(q5) * sin(q6) + cos(q2) * cos(q4) * cos(q5) * sin(q3) * sin(q6) + cos(q3) * cos(q4) * cos(q5) * sin(q2) * sin(q6)), 0, 0, 0, g * (cos(q2) * cos(q3) * cos(q6) * sin(q5) - cos(q2) * sin(q3) * sin(q4) * sin(q6) - cos(q3) * sin(q2) * sin(q4) * sin(q6) - cos(q6) * sin(q2) * sin(q3) * sin(q5) + cos(q2) * cos(q4) * cos(q5) * cos(q6) * sin(q3) + cos(q3) * cos(q4) * cos(q5) * cos(q6) * sin(q2)), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        return R_mat


class MTM_MLSE4POL():
    def __init__(self):
        self.g = 9.81
        self.param_vec = np.array([ -0.3049,
        0.1305,
        0.0280,
        0.0127,
        0.0002,
        0.0209,

    -0.0105,
    -0.0018,
    0.0046,
    0.0001,
    0.0335,
    0.0635,
    0.0483,
-0.0070,
-0.0936,
-1.0210,
3.4479,
0,
0,
0,
0.0234,
0.0891,
0.0125,
-0.0311,
0.1426,
0.0492,
0.0415,
-0.0234,
-0.0009,
0.0019,
0.0590,
-0.0968,
-0.0256,
0.0199,
-0.0014,
0.0008,
0.0022,
0.0022,
-0.0017,
0.0177,
0.0120,
0.0683,
0.0068,
0.1208,
0.0699,
-1.0946,
3.4335,
0,
0,
0,
0.0004,
0.0589,
-0.0310,
0.0719,
0.3217,
-0.0508,
0.0500,
-0.0194,
-0.0087,
0.0038,
0.0398,
-0.0951,
-0.0237,
0.0196,
-0.0016,
-0.0010,
-0.0033,
-0.0100,
0.0034,
0.0097]).reshape((70,1))

    def regressor_pos(self, q1, q2, q3, q4, q5, q6):
            g = self.g
            R_mat = [[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 1, q1, q1**2, q1**3, q1**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ g*sin(q2), g*cos(q2), g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0,  0,    0,    0,    0, 1, q2, q2**2, q2**3, q2**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0, g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q3, q3**2, q3**3, q3**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                               -g*sin(q2 + q3)*sin(q4),                               -g*sin(q2 + q3)*cos(q4),                                                                                                g*sin(q2 + q3)*sin(q4)*sin(q5),                                                                                               -g*sin(q2 + q3)*cos(q5)*sin(q4),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*sin(q6) + cos(q5)*cos(q6)*sin(q4)),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*cos(q6) - cos(q5)*sin(q4)*sin(q6)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q4, q4**2, q4**3, q4**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,    -g*(cos(q2)*cos(q3)*sin(q5) - sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q2)),    -g*(cos(q5)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5) + cos(q2)*cos(q4)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*sin(q2)*sin(q5)),                                                                             g*(cos(q5)*cos(q6)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5)*cos(q6) + cos(q2)*cos(q4)*cos(q6)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*cos(q6)*sin(q2)*sin(q5)),                                                                            -g*(cos(q5)*sin(q2)*sin(q3)*sin(q6) - cos(q2)*cos(q3)*cos(q5)*sin(q6) + cos(q2)*cos(q4)*sin(q3)*sin(q5)*sin(q6) + cos(q3)*cos(q4)*sin(q2)*sin(q5)*sin(q6)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q5, q5**2, q5**3, q5**4, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,         g*(cos(q2)*cos(q6)*sin(q3)*sin(q4) + cos(q3)*cos(q6)*sin(q2)*sin(q4) + cos(q2)*cos(q3)*sin(q5)*sin(q6) - sin(q2)*sin(q3)*sin(q5)*sin(q6) + cos(q2)*cos(q4)*cos(q5)*sin(q3)*sin(q6) + cos(q3)*cos(q4)*cos(q5)*sin(q2)*sin(q6)),         g*(cos(q2)*cos(q3)*cos(q6)*sin(q5) - cos(q2)*sin(q3)*sin(q4)*sin(q6) - cos(q3)*sin(q2)*sin(q4)*sin(q6) - cos(q6)*sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*cos(q6)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*cos(q6)*sin(q2)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q6, q6**2, q6**3, q6**4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

            return R_mat

    def regressor_neg(self, q1, q2, q3, q4, q5, q6):
        g = self.g
        R_mat = np.array([[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, q1, q1**2, q1**3, q1**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[ g*sin(q2), g*cos(q2), g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 1, q2, q2**2, q2**3, q2**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0, g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q3, q3**2, q3**3, q3**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                               -g*sin(q2 + q3)*sin(q4),                               -g*sin(q2 + q3)*cos(q4),                                                                                                g*sin(q2 + q3)*sin(q4)*sin(q5),                                                                                               -g*sin(q2 + q3)*cos(q5)*sin(q4),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*sin(q6) + cos(q5)*cos(q6)*sin(q4)),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*cos(q6) - cos(q5)*sin(q4)*sin(q6)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q4, q4**2, q4**3, q4**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,    -g*(cos(q2)*cos(q3)*sin(q5) - sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q2)),    -g*(cos(q5)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5) + cos(q2)*cos(q4)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*sin(q2)*sin(q5)),                                                                             g*(cos(q5)*cos(q6)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5)*cos(q6) + cos(q2)*cos(q4)*cos(q6)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*cos(q6)*sin(q2)*sin(q5)),                                                                            -g*(cos(q5)*sin(q2)*sin(q3)*sin(q6) - cos(q2)*cos(q3)*cos(q5)*sin(q6) + cos(q2)*cos(q4)*sin(q3)*sin(q5)*sin(q6) + cos(q3)*cos(q4)*sin(q2)*sin(q5)*sin(q6)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q5, q5**2, q5**3, q5**4, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,         g*(cos(q2)*cos(q6)*sin(q3)*sin(q4) + cos(q3)*cos(q6)*sin(q2)*sin(q4) + cos(q2)*cos(q3)*sin(q5)*sin(q6) - sin(q2)*sin(q3)*sin(q5)*sin(q6) + cos(q2)*cos(q4)*cos(q5)*sin(q3)*sin(q6) + cos(q3)*cos(q4)*cos(q5)*sin(q2)*sin(q6)),         g*(cos(q2)*cos(q3)*cos(q6)*sin(q5) - cos(q2)*sin(q3)*sin(q4)*sin(q6) - cos(q3)*sin(q2)*sin(q4)*sin(q6) - cos(q6)*sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*cos(q6)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*cos(q6)*sin(q2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q6, q6**2, q6**3, q6**4],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0]])

        return R_mat
    
    def predict(self, input_mat):
        output_mat = np.zeros((input_mat.shape[0], 5))
        for i in range(input_mat.shape[0]):
            q2 = input_mat[i,0]
            q3 = input_mat[i,1]
            q4 = input_mat[i,2]
            q5 = input_mat[i,3]
            q6 = input_mat[i,4]
            R_mat = (self.regressor_pos(0, q2, q3, q4, q5, q6) + self.regressor_neg(0, q2, q3, q4, q5, q6))/2
            tor = R_mat.dot(self.param_vec).reshape(7)
            output_mat[i,:] = tor[1:-1]
        return output_mat
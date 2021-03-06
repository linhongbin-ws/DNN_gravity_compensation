function [abs_rms, rel_rms] = base_line_MLSE4POL(input_mat, output_mat)
    input_mat = input_mat.';
    output_mat = output_mat.';
    g = 9.81;
    output_hat_mat = [];
    dynamic_vec = get_dynamic_vec()
    for i = 1:size(input_mat,2)
        q1 = input_mat(1,i);
        q2 = input_mat(2,i);
        q3 = input_mat(3,i);
        q4 = input_mat(4,i);
        q5 = input_mat(5,i);
        q6 = input_mat(6,i);
        torque_pos = analytical_regressor_pos_mat(g, q1, q2, q3, q4, q5, q6)*dynamic_vec;
        torque_neg = analytical_regressor_neg_mat(g, q1, q2, q3, q4, q5, q6)*dynamic_vec;
        output_hat_mat = [output_hat_mat, (torque_pos+torque_neg)/2];
    end
    output_hat_mat = output_hat_mat(1:6,:);
    e_mat = output_mat-output_hat_mat;
    abs_rms = sqrt(mean(e_mat.^2, 2));
    rel_rms = abs_rms./sqrt(mean(output_hat_mat.^2, 2));
end

function dynamic_vec = get_dynamic_vec()
  dynamic_vec =[-0.3049
    0.1305
    0.0280
    0.0127
    0.0002
    0.0209
   -0.0105
   -0.0018
    0.0046
    0.0001
    0.0335
    0.0635
    0.0483
   -0.0070
   -0.0936
   -1.0210
    3.4479
    0.0234
    0.0891
    0.0125
   -0.0311
    0.1426
    0.0492
    0.0415
   -0.0234
   -0.0009
    0.0019
    0.0590
   -0.0968
   -0.0256
    0.0199
   -0.0014
    0.0008
    0.0022
    0.0022
   -0.0017
    0.0177
    0.0120
    0.0683
    0.0068
    0.1208
    0.0699
   -1.0946
    3.4335
    0.0004
    0.0589
   -0.0310
    0.0719
    0.3217
   -0.0508
    0.0500
   -0.0194
   -0.0087
    0.0038
    0.0398
   -0.0951
   -0.0237
    0.0196
   -0.0016
   -0.0010
   -0.0033
   -0.0100
    0.0034
    0.0097];
end



function Regressor_Matrix_Pos = analytical_regressor_pos_mat(g,q1,q2,q3,q4,q5,q6)
%ANALYTICAL_REGRESSOR_POS_MAT
%    REGRESSOR_MATRIX_POS = ANALYTICAL_REGRESSOR_POS_MAT(G,Q1,Q2,Q3,Q4,Q5,Q6)

%    This function was generated by the Symbolic Math Toolbox version 8.3.
%    04-Oct-2019 13:46:01

t2 = cos(q2);
t3 = cos(q3);
t4 = cos(q4);
t5 = cos(q5);
t6 = cos(q6);
t7 = sin(q2);
t8 = sin(q3);
t9 = sin(q4);
t10 = sin(q5);
t11 = sin(q6);
t12 = q2+q3;
t13 = sin(t12);
t14 = g.*t2.*t3;
t15 = g.*t2.*t8;
t16 = g.*t3.*t7;
t17 = g.*t7.*t8;
t18 = t4.*t14;
t19 = t9.*t14;
t20 = t5.*t15;
t21 = t5.*t16;
t22 = t4.*t17;
t23 = t10.*t15;
t24 = t10.*t16;
t25 = t9.*t17;
t26 = -t15;
t27 = -t16;
t28 = -t17;
t29 = t5.*t18;
t30 = t10.*t18;
t31 = t6.*t19;
t32 = t5.*t22;
t33 = t11.*t19;
t34 = t6.*t23;
t35 = t6.*t24;
t36 = t10.*t22;
t37 = t6.*t25;
t38 = t11.*t23;
t39 = t11.*t24;
t40 = -t19;
t41 = -t20;
t42 = -t21;
t43 = t11.*t25;
t44 = -t22;
t45 = -t23;
t46 = -t24;
t59 = t14+t28;
t60 = t26+t27;
t47 = t6.*t32;
t48 = t11.*t32;
t49 = -t30;
t50 = -t32;
t51 = -t37;
t52 = -t38;
t53 = -t39;
t54 = -t43;
t55 = t6.*t29;
t56 = t11.*t29;
t61 = t18+t44;
t62 = t25+t40;
t57 = -t48;
t58 = -t55;
t63 = t36+t41+t42+t49;
t64 = t29+t45+t46+t50;
t65 = t33+t34+t35+t47+t54+t58;
t66 = t31+t51+t52+t53+t56+t57;
Regressor_Matrix_Pos = reshape([0.0,g.*t7,0.0,0.0,0.0,0.0,0.0,0.0,g.*t2,0.0,0.0,0.0,0.0,0.0,0.0,t59,t59,0.0,0.0,0.0,0.0,0.0,t60,t60,0.0,0.0,0.0,0.0,0.0,t61,t61,-g.*t9.*t13,0.0,0.0,0.0,0.0,t62,t62,-g.*t4.*t13,0.0,0.0,0.0,0.0,t63,t63,g.*t9.*t10.*t13,-g.*(t2.*t3.*t10-t7.*t8.*t10+t2.*t4.*t5.*t8+t3.*t4.*t5.*t7),0.0,0.0,0.0,t64,t64,-g.*t5.*t9.*t13,-g.*(-t2.*t3.*t5+t5.*t7.*t8+t2.*t4.*t8.*t10+t3.*t4.*t7.*t10),0.0,0.0,0.0,t65,t65,g.*t13.*(t4.*t11+t5.*t6.*t9),g.*(-t2.*t3.*t5.*t6+t5.*t6.*t7.*t8+t2.*t4.*t6.*t8.*t10+t3.*t4.*t6.*t7.*t10),g.*(t2.*t6.*t8.*t9+t3.*t6.*t7.*t9+t2.*t3.*t10.*t11-t7.*t8.*t10.*t11+t2.*t4.*t5.*t8.*t11+t3.*t4.*t5.*t7.*t11),0.0,0.0,t66,t66,g.*t13.*(t4.*t6-t5.*t9.*t11),-g.*(-t2.*t3.*t5.*t11+t5.*t7.*t8.*t11+t2.*t4.*t8.*t10.*t11+t3.*t4.*t7.*t10.*t11),g.*(t2.*t3.*t6.*t10-t2.*t8.*t9.*t11-t3.*t7.*t9.*t11-t6.*t7.*t8.*t10+t2.*t4.*t5.*t6.*t8+t3.*t4.*t5.*t6.*t7),0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q1,0.0,0.0,0.0,0.0,0.0,0.0,q1.^2,0.0,0.0,0.0,0.0,0.0,0.0,q1.^3,0.0,0.0,0.0,0.0,0.0,0.0,q1.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q3,0.0,0.0,0.0,0.0,0.0,0.0,q3.^2,0.0,0.0,0.0,0.0,0.0,0.0,q3.^3,0.0,0.0,0.0,0.0,0.0,0.0,q3.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q4,0.0,0.0,0.0,0.0,0.0,0.0,q4.^2,0.0,0.0,0.0,0.0,0.0,0.0,q4.^3,0.0,0.0,0.0,0.0,0.0,0.0,q4.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q5,0.0,0.0,0.0,0.0,0.0,0.0,q5.^2,0.0,0.0,0.0,0.0,0.0,0.0,q5.^3,0.0,0.0,0.0,0.0,0.0,0.0,q5.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q6,0.0,0.0,0.0,0.0,0.0,0.0,q6.^2,0.0,0.0,0.0,0.0,0.0,0.0,q6.^3,0.0,0.0,0.0,0.0,0.0,0.0,q6.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[7,64]);
end

function Regressor_Matrix_Neg = analytical_regressor_neg_mat(g,q1,q2,q3,q4,q5,q6)
%ANALYTICAL_REGRESSOR_NEG_MAT
%    REGRESSOR_MATRIX_NEG = ANALYTICAL_REGRESSOR_NEG_MAT(G,Q1,Q2,Q3,Q4,Q5,Q6)

%    This function was generated by the Symbolic Math Toolbox version 8.3.
%    04-Oct-2019 13:46:02

t2 = cos(q2);
t3 = cos(q3);
t4 = cos(q4);
t5 = cos(q5);
t6 = cos(q6);
t7 = sin(q2);
t8 = sin(q3);
t9 = sin(q4);
t10 = sin(q5);
t11 = sin(q6);
t12 = q2+q3;
t13 = sin(t12);
t14 = g.*t2.*t3;
t15 = g.*t2.*t8;
t16 = g.*t3.*t7;
t17 = g.*t7.*t8;
t18 = t4.*t14;
t19 = t9.*t14;
t20 = t5.*t15;
t21 = t5.*t16;
t22 = t4.*t17;
t23 = t10.*t15;
t24 = t10.*t16;
t25 = t9.*t17;
t26 = -t15;
t27 = -t16;
t28 = -t17;
t29 = t5.*t18;
t30 = t10.*t18;
t31 = t6.*t19;
t32 = t5.*t22;
t33 = t11.*t19;
t34 = t6.*t23;
t35 = t6.*t24;
t36 = t10.*t22;
t37 = t6.*t25;
t38 = t11.*t23;
t39 = t11.*t24;
t40 = -t19;
t41 = -t20;
t42 = -t21;
t43 = t11.*t25;
t44 = -t22;
t45 = -t23;
t46 = -t24;
t59 = t14+t28;
t60 = t26+t27;
t47 = t6.*t32;
t48 = t11.*t32;
t49 = -t30;
t50 = -t32;
t51 = -t37;
t52 = -t38;
t53 = -t39;
t54 = -t43;
t55 = t6.*t29;
t56 = t11.*t29;
t61 = t18+t44;
t62 = t25+t40;
t57 = -t48;
t58 = -t55;
t63 = t36+t41+t42+t49;
t64 = t29+t45+t46+t50;
t65 = t33+t34+t35+t47+t54+t58;
t66 = t31+t51+t52+t53+t56+t57;
Regressor_Matrix_Neg = reshape([0.0,g.*t7,0.0,0.0,0.0,0.0,0.0,0.0,g.*t2,0.0,0.0,0.0,0.0,0.0,0.0,t59,t59,0.0,0.0,0.0,0.0,0.0,t60,t60,0.0,0.0,0.0,0.0,0.0,t61,t61,-g.*t9.*t13,0.0,0.0,0.0,0.0,t62,t62,-g.*t4.*t13,0.0,0.0,0.0,0.0,t63,t63,g.*t9.*t10.*t13,-g.*(t2.*t3.*t10-t7.*t8.*t10+t2.*t4.*t5.*t8+t3.*t4.*t5.*t7),0.0,0.0,0.0,t64,t64,-g.*t5.*t9.*t13,-g.*(-t2.*t3.*t5+t5.*t7.*t8+t2.*t4.*t8.*t10+t3.*t4.*t7.*t10),0.0,0.0,0.0,t65,t65,g.*t13.*(t4.*t11+t5.*t6.*t9),g.*(-t2.*t3.*t5.*t6+t5.*t6.*t7.*t8+t2.*t4.*t6.*t8.*t10+t3.*t4.*t6.*t7.*t10),g.*(t2.*t6.*t8.*t9+t3.*t6.*t7.*t9+t2.*t3.*t10.*t11-t7.*t8.*t10.*t11+t2.*t4.*t5.*t8.*t11+t3.*t4.*t5.*t7.*t11),0.0,0.0,t66,t66,g.*t13.*(t4.*t6-t5.*t9.*t11),-g.*(-t2.*t3.*t5.*t11+t5.*t7.*t8.*t11+t2.*t4.*t8.*t10.*t11+t3.*t4.*t7.*t10.*t11),g.*(t2.*t3.*t6.*t10-t2.*t8.*t9.*t11-t3.*t7.*t9.*t11-t6.*t7.*t8.*t10+t2.*t4.*t5.*t6.*t8+t3.*t4.*t5.*t6.*t7),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q1,0.0,0.0,0.0,0.0,0.0,0.0,q1.^2,0.0,0.0,0.0,0.0,0.0,0.0,q1.^3,0.0,0.0,0.0,0.0,0.0,0.0,q1.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q3,0.0,0.0,0.0,0.0,0.0,0.0,q3.^2,0.0,0.0,0.0,0.0,0.0,0.0,q3.^3,0.0,0.0,0.0,0.0,0.0,0.0,q3.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q4,0.0,0.0,0.0,0.0,0.0,0.0,q4.^2,0.0,0.0,0.0,0.0,0.0,0.0,q4.^3,0.0,0.0,0.0,0.0,0.0,0.0,q4.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q5,0.0,0.0,0.0,0.0,0.0,0.0,q5.^2,0.0,0.0,0.0,0.0,0.0,0.0,q5.^3,0.0,0.0,0.0,0.0,0.0,0.0,q5.^4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,q6,0.0,0.0,0.0,0.0,0.0,0.0,q6.^2,0.0,0.0,0.0,0.0,0.0,0.0,q6.^3,0.0,0.0,0.0,0.0,0.0,0.0,q6.^4,0.0],[7,64]);
end
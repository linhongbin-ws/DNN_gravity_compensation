load('../data/Acrobot_sim_25/Acrobot_sim_25.mat')

R_mat = [];
tau_vec = [];
for i = 1:size(input_mat,1)
    q1 = input_mat(i,1);
    q2 = input_mat(i,2);
    tau_vec = [tau_vec;output_mat(i,1);output_mat(i,2)];
    R_mat = [R_mat;Regressor(q1,q2)];
end

 beta_vec = pinv(R_mat)*tau_vec;


train_input_mat = input_mat;
train_output_mat = output_mat;

load('../data/Acrobot_sim_1156/Acrobot_sim_1156.mat')
test_output_mat = [];
for i = 1:size(input_mat,1)
    q1 = input_mat(i,1);
    q2 = input_mat(i,2);
    tau =Regressor(q1,q2)*beta_vec;
    test_output_mat = [test_output_mat;tau.'];
end
test_input_mat = input_mat;

save('../figure/bp_train_8/result.mat')

function R = Regressor(q1,q2)
    s1 =sin(q1);
    s12 = sin(q1+q2);
    R = [s1, s12;0, s12];
end
load('../figure/bp_train_8/result.mat')
target_mat = [];
for i=1:size(test_input_mat,1)
    q1 = input_mat(i,1);
    q2 = input_mat(i,2);
    G = Acrobot_gravity(q1, q2, 0, 0);
    target_mat = [target_mat;G.'];
end

rel_rms = sqrt(sum((test_output_mat-target_mat).^2)./sum((target_mat).^2))
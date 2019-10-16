net_list = {'ReLuNet.mat','SigmoidNet.mat','SinNet.mat','Lagrangian_SinNet.mat'}
N_arr = [5,8,15];
std_arr = [1, 5, 9];

result_list ={};
for j=1:size(std_arr,2)
    for k=1:size(N_arr,2) 
        result_mat = [];
        for i=1:numel(net_list)
            file_name=sprintf('N%d_std%d/result/%s', N_arr(k), std_arr(j), net_list{i});
            rel_rms_vec = cal_rms(file_name);
            result_mat = [result_mat,rel_rms_vec]; 
        end
        result_list{end+1} = result_mat;
    end
end



function rel_rms_vec = cal_rms(load_file)
    load(load_file)
    target_mat = [];
    for i=1:size(test_input_mat,1)
        q1 = test_input_mat(i,1);
        q2 = test_input_mat(i,2);
        G = Acrobot_gravity(q1, q2, 0, 0);
        target_mat = [target_mat;G.'];
    end
    rel_rms = sqrt(sum((test_output_mat-target_mat).^2)./sum((target_mat).^2));
    rel_rms_vec = rel_rms.';
end
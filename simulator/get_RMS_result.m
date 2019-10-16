net_list = {'ReLuNet.mat','SigmoidNet.mat','SinNet.mat','Lagrangian_SinNet.mat','dynamic_model.mat'}
legend_list = {'ReLu Net','Sigmoid Net','Sin Net','Lagrangian-Sin Net', 'Dynamic Model'}
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

plot_rms(1, 1, legend_list,N_arr, result_list(1:3))
plot_rms(2, 1, legend_list,N_arr, result_list(4:6))
plot_rms(3, 1, legend_list,N_arr, result_list(7:9))
plot_rms(4, 2, legend_list,N_arr, result_list(1:3))
plot_rms(5, 2, legend_list,N_arr, result_list(4:6))
plot_rms(6, 2, legend_list,N_arr, result_list(7:9))

function plot_rms(plt_idx, jnt_no, legend_list,N_arr, result_list)
    linewidth = 3;
    mksize =12;
    net_result_list ={};
    for j=1:size(result_list{1},2)
        result_mat = []
        for i=1:numel(result_list)
            result_mat = [result_mat, result_list{i}(:,j)]
        end
        net_result_list{end+1} = result_mat;
    end
    figure(plt_idx)
    hold on
    fntsize = 20;
    p1 = plot(N_arr, net_result_list{1}(jnt_no,:),'-x','LineWidth',linewidth,'MarkerSize',mksize);
    p2 = plot(N_arr, net_result_list{2}(jnt_no,:),'-x','LineWidth',linewidth,'MarkerSize',mksize);
    p3 = plot(N_arr, net_result_list{3}(jnt_no,:),'-x','LineWidth',linewidth,'MarkerSize',mksize);
    p4 = plot(N_arr, net_result_list{4}(jnt_no,:),'-x','LineWidth',linewidth,'MarkerSize',mksize);
    p5 = plot(N_arr, net_result_list{5}(jnt_no,:),'-x','LineWidth',linewidth,'MarkerSize',mksize);
    hold off
    set(gca,'FontSize',fntsize);
    legend([p1,p2,p3,p4, p5], legend_list,'location','best');
    xlabel('$N$','interpreter','latex','FontSize',fntsize)
    ylabel(sprintf("$\\epsilon_{RMS_%d} (\\%%)$", jnt_no),'interpreter','latex','FontSize',fntsize)
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
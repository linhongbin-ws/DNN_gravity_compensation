jnt_sample_num_arr = [2,3,4,5,6,7,8,9,10,12,15,17,20];
%jnt_sample_num_arr = [34]
std_arr = [1];
for i = 1:size(jnt_sample_num_arr,2)
    for j=1:size(std_arr,2)
        generate_data(jnt_sample_num_arr(i), std_arr(j));
    end
end

function generate_data(jnt_sample_num, std)
jnt_range = linspace(-pi,pi,jnt_sample_num);
[X, Y] = meshgrid(jnt_range);
Z1_noise = zeros(size(X));
Z2_noise = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        G_noise = Acrobot_gravity(X(i,j), Y(i,j), std, std);
        Z1_noise(i,j) = G_noise(1);
        Z2_noise(i,j) = G_noise(2);
    end
end

input_mat = [X(:), Y(:)];
output_mat = [Z1_noise(:), Z2_noise(:)];

mkdir(sprintf('N%d_std%d/data', jnt_sample_num, std))
save(sprintf('N%d_std%d/data/N%d_std%d.mat', jnt_sample_num, std,jnt_sample_num, std),'input_mat','output_mat')
end
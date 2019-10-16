jnt_sample_num = 8;
std = 9;
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

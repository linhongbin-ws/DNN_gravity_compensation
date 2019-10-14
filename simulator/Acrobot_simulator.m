jnt_sample_num = 15;
jnt_range = linspace(-pi,pi,jnt_sample_num);
[X, Y] = meshgrid(jnt_range);
Z1 = zeros(size(X));
Z2 = zeros(size(X));
Z1_noise = zeros(size(X));
Z2_noise = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        G = Acrobot_gravity(X(i,j), Y(i,j), false);
        G_noise = Acrobot_gravity(X(i,j), Y(i,j), true);
        Z1(i,j) = G(1);
        Z2(i,j) = G(2);
        Z1_noise(i,j) = G_noise(1);
        Z2_noise(i,j) = G_noise(2);
    end
end

input_mat = [X(:), Y(:)];
output_mat = [Z1_noise(:), Z2_noise(:)];

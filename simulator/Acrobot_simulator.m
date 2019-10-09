delta = 3;
fntsize = 20;
jnt_range = -180:delta:180;
[X, Y] = meshgrid(jnt_range);
Z1 = zeros(size(X));
Z2 = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        G = acrobot_gravity(deg2rad(X(i,j)), deg2rad(Y(i,j)));
        Z1(i,j) = G(1);
        Z2(i,j) = G(2);
    end
end
figure(1)
hold on
surfplot = surf(X,Y,Z1);
Z1_noise = Z1+randn(size(Z1));
markersize = 5;
splot = scatter3(X(:),Y(:), Z1_noise(:),markersize,'filled');
splot.MarkerFaceAlpha = 0.2;
alpha 0.7
hold off
view(3)
xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
zlabel('$\tau_1(N.m)$','interpreter','latex','FontSize', fntsize)
ticks = [-180 -90 0 90 180]
set(gca,'FontSize',fntsize);
xticks(ticks)
yticks(ticks)
ticks = [min(Z1,[],'all'), 0, max(Z1,[],'all')];
zticks(ticks)
title('Measuring and ground-truth gravity for Joint 1 of Acrobot')
legend([surfplot, splot], {'Ground-truth gravity', 'Measuring gravity'});


figure(2)
hold on
surfplot = surf(X,Y,Z2);
Z2_noise = Z2+randn(size(Z2));
markersize = 5;
splot = scatter3(X(:),Y(:), Z2_noise(:),markersize,'filled');
splot.MarkerFaceAlpha = 0.2;
alpha 0.7
hold off
view(3)
xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
zlabel('$\tau_2(N.m)$','interpreter','latex','FontSize', fntsize)
ticks = [-180 -90 0 90 180]
set(gca,'FontSize',fntsize);
xticks(ticks)
yticks(ticks)
ticks = [min(Z2,[],'all'), 0, max(Z2,[],'all')];
zticks(ticks)
title('Measuring and ground-truth gravity for Joint 2 of Acrobot')
legend([surfplot, splot], {'Ground-truth gravity', 'Measuring gravity'});


function G = acrobot_gravity(q1, q2)
    l1  = 1;
    l2  = 2;
    lc1 = 0.5; % COM of link 1
    lc2 = 1; % COM of link 2
    m1  = 1;
    m2  = 1;
    g = 9.81;
    s = [sin(q1), sin(q2)];
    s12 = sin(q1+q2);
    G1 = g*(m1*lc1*s(1) + m2*(l1*s(1)+lc2*s12));
    G2 = g*m2*lc2*s12;
    G  = [ G1; G2 ];
end
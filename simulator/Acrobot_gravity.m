function G = Acrobot_gravity(q1, q2, is_noise)
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
    std1 = 1;
    std2 = 1;
    if is_noise
        G1 = G1 + randn(1)*std1;
        G2 = G2 + randn(1)*std2;
    end
    G  = [ G1; G2 ];
end
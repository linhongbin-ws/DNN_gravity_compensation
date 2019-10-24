function torques = extDisturb(q1, q2)
    torques = zeros(2,1);
    torques(1) = 2*q1^2+2*q2^2+0.1*q1*q2;
    torques(2) = 2*q1^2+2*q2^2+0.1*q1*q2;
end
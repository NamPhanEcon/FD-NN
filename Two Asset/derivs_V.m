function [V1_B,V1_A] = derivs_V(parameters,az,B,A,Y,Z)


V = model_DL_V(parameters,az,B,A,Y,Z);

V_temp = dlgradient(sum(V,'all'), [B; A]);

V1_B = V_temp(1,:);
V1_A = V_temp(2,:);

end
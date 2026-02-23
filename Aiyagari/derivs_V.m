function [V_A] = derivs_V(parameters,az,AZ)
V = model_DL_C(parameters,az,AZ);
V_tempA = dlgradient(sum(V,'all'),AZ,'EnableHigherDerivatives',true); 
V_A = V_tempA(1,:);
end
function [K_T,Asim_T_DL] = aiyagari_transitional_DL_C(Asim_init,z_t,parameters,T,Delta_sim,N_draw,mu_C,std_C)

alpha = 1/3; 
TFP = 1; delta = 0.1; 


A_next = Asim_init ; 
Asim_T_DL = zeros(N_draw,T); K_T= zeros(T,1);

for t = 1:T
A_sim = A_next; 

A_in = [];
A_in = dlarray(A_sim,'TCB');
Z_sim = z_t(:,t); 
Z_in = dlarray(Z_sim,'TCB');

AZ = cat(1,A_in,Z_in);
mean_A = mean(A_sim); mean_Z = mean(Z_sim);
r =  alpha     * TFP * mean_A.^(alpha-1) * mean_Z.^(1-alpha) -delta ;%interest rates
w = (1-alpha) * TFP * mean_A.^(alpha) * mean_Z.^(-alpha) ;          %wages
c = model_DL_C(parameters,AZ,AZ); 

c_dl = gather(extractdata(squeeze(c)))*std_C + mu_C  ;  
Adot_each = r*A_sim + w*Z_sim - c_dl; 

A_next = Adot_each * Delta_sim + A_sim ; %update: 

K_T(t)       = mean(A_sim);

Asim_T_DL(:,t) = A_sim; 
end
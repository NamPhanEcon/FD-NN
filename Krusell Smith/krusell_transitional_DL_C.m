function [K_T,Asim_T_DL] = krusell_transitional_DL_C(A_in,z_T,H_in,parameters,T,Delta_sim,N_in,mean_C,std_C)

lam_1 = 0.4;
lam_2 = 0.4; 
y1 = 0.3; 
y2 = 1 + (lam_2/lam_1*(1-y1)); 

alpha = 1/3; 
delta = 0.1; 

A_next = A_in ; 
Asim_T_DL = zeros(N_in,T); K_T= zeros(T,1);

c_cell = cell(T,1); 
r_t = zeros(T,1) ;
w_t = zeros(T,1) ;

for t = 1:T
A_sim = A_next; 
z_in = z_T(:,t); 
A_in = [];
A_in = dlarray(A_sim,'TCB');
Z_sim = z_in; 
Z_in = dlarray(Z_sim,'TCB');

H_sim = repmat(H_in(t),[N_in 1]) ; 

H_in_dl = dlarray(H_sim,'TCB'); 

az = cat(1,A_in,Z_in,H_in_dl);
AZ = cat(1,A_in,Z_in);

mean_A = mean(A_sim); mean_Z = mean(Z_sim);
r =  alpha     * exp(H_in(t)) * mean_A.^(alpha-1) * mean_Z.^(1-alpha) -delta ;  %interest rates
w = (1-alpha) * exp(H_in(t))* mean_A.^(alpha) * mean_Z.^(-alpha) ;           %wages

r_t(t) = alpha     * exp(H_in(t)) * mean_A.^(alpha-1) * mean_Z.^(1-alpha) -delta; 
w_t(t) = (1-alpha) * exp(H_in(t)) * mean_A.^(alpha) * mean_Z.^(-alpha) ; 

c = model_DL_C(parameters,az,AZ); 

c_dl = gather(extractdata(squeeze(c))) ;  

c_dl = c_dl*std_C + mean_C; 
c_cell{t} = c_dl; 
Adot_each = r*A_sim + w*Z_sim - c_dl; 

A_next = Adot_each * Delta_sim + A_sim ; %update: 

K_T(t)       = mean(A_sim);

Asim_T_DL(:,t) = A_sim; 

end

end
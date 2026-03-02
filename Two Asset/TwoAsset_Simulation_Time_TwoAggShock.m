function [K_T,A_vec_t,B_vec_t,C_vec_t,D_vec_t,Adot_T,Bdot_T] = ...
    TwoAsset_Simulation_Time_TwoAggShock(B_in,A_in,y_T,z_T,...
     H_T,param,N_draw,parameters,dt,T) 


TFP_mat = [param.z_low,param.z_mid]; 
adjusted_ratio = (param.frequency_recession/(1-param.frequency_recession)); 

y = param.z; 
Delta_sim = dt; 
a = param.a; 
b = param.b; 
mean_z2 = param.mean_z2; 
target_u_mid = param.target_u_mid;
target_u_recession = param.target_u_recession;

y = param.z; 

tau_param = param.tau_param; 
chi0 = param.chi0 ;
chi1 = param.chi1; 
chi2 = param.chi2; 
abar = param.abar;

unemp_benefit = param.unemp_benefit; 
transfer = param.transfer_ss ; 

% J           = param.J_a; 
% I           = param.I;

tauc        = param.tauc; 
amax_in     = param.a(end); 

alpha       = param.alpha; 
depreciation_rate       = param.depreciation_rate; 

A_vec   = A_in;
B_vec   = B_in; 

K_T = zeros(T,1);
A_T = zeros(T,1);

C_T = zeros(T,1);
D_T = zeros(T,1);

amin_simulate = 1e-6; 
bmin_simulate = b(1)*0.99999; 
bmax_simulate = b(end)*0.99999; 

sum_bind_a = zeros(T,1); 
sum_bind_b = zeros(T,1); 

A_vec_t = zeros(N_draw,T);
B_vec_t = zeros(N_draw,T);
C_vec_t = zeros(N_draw,1);
D_vec_t = zeros(N_draw,1);

Adot_T = zeros(N_draw,T);
Bdot_T = zeros(N_draw,T);

transfer_prop = 0; 

 for t = 1:T

z_in    = z_T(:,t) ;
y_index = y_T(:,t) ;
y_in    = y(y_index) ; 

A_sim = A_vec; 
B_sim = B_vec; 

A_dl = dlarray(A_sim,'TCB');
B_dl = dlarray(B_sim,'TCB');

Z_sim = z_in; 
Z_dl = dlarray(Z_sim,'TCB');

Y_sim = y_in; 
Y_dl = dlarray(Y_sim,'TCB'); 

H_dl_index = dlarray(repmat(H_T(t),[N_draw 1]),'TCB') ; 

az = cat(1,B_dl,A_dl,Y_dl,Z_dl,H_dl_index);
AZ = cat(1,B_dl,A_dl,Y_dl,Z_dl);

C = model_DL_C(parameters.netParamsC,az,AZ);
D = model_DL_C(parameters.netParamsD,az,AZ);

consumption = extractdata(gather(squeeze(C))) ; 
wdl = extractdata(gather(squeeze(D))) ; 

% --- MODEL OBJECTS 
Ksim = mean(A_vec) + mean(B_vec) ; 

L =  mean_z2*(1-target_u_recession)*(H_T(t)==1) + mean_z2*(1-target_u_mid)*(H_T(t)==2) ; 

TFP     = TFP_mat(H_T(t)) ; 
ra      =  alpha     * TFP * Ksim.^(alpha-1) * L^(1-alpha)  - depreciation_rate ;
w       =  (1-alpha) * TFP* Ksim.^(alpha)*L^(-alpha)  ;

ra_vec = ra*(1 - (1.33.*amax_in./A_vec).^(1-tauc)) ; 

rb_pos = ra - (param.ra-param.rb_pos) ;
rb_neg = param.rb_neg ; 
rb_vec = rb_pos*(B_vec>=0) + (rb_neg)*(B_vec<0) ;

constraint_a = ((wdl+ra_vec.*A_vec)*Delta_sim + A_vec)< 0 ; 

wdl_fix = wdl.*(constraint_a==0) + ((amin_simulate - A_vec)/Delta_sim - ra_vec.*A_vec).*(constraint_a==1).*(A_vec>0) ...
  + (constraint_a==1).*(A_vec==0).*0; 

transaction_cost_vec = two_asset_kinked_cost_fortran(wdl_fix,A_vec,chi0,chi1,chi2,abar) ; 

sum_bind_a(t) = sum(constraint_a) ; 


constraint_b = ((rb_vec) .* B_vec + (z_in==2).*(1-tau_param).*w.*y_in + (z_in==1).*unemp_benefit + transfer ...
+ (transfer*transfer_prop*(H_T(t)==1) - transfer*transfer_prop*(H_T(t)==2)*adjusted_ratio) ...
- wdl_fix - consumption ...
- transaction_cost_vec)*Delta_sim + B_vec < bmin_simulate ;  

constraint_b_upper = ((rb_vec) .* B_vec + (z_in==2).*(1-tau_param).*w.*y_in + (z_in==1).*unemp_benefit + transfer ...
+ (transfer*transfer_prop*(H_T(t)==1) - transfer*transfer_prop*(H_T(t)==2)*adjusted_ratio) ...
- wdl_fix - consumption - transaction_cost_vec)*Delta_sim + B_vec > bmax_simulate ; 

sum_bind_b(t) = sum(constraint_b) ; 

consumption_fix = consumption.*(constraint_b==0).*(constraint_b_upper==0) + ...
    (constraint_b==1).*(-(bmin_simulate - B_vec)*Delta_sim + B_vec.*rb_vec ...
    +  (z_in==2).*(1-tau_param).*w.*y_in + (z_in==1).*unemp_benefit + transfer ...
    + (transfer*transfer_prop*(H_T(t)==1) - transfer*transfer_prop*(H_T(t)==2)*adjusted_ratio) ...
    - wdl_fix -  transaction_cost_vec) + ...
(constraint_b_upper==1).*(-(bmax_simulate - B_vec)*Delta_sim + B_vec.*rb_vec ...
    +  (z_in==2).*(1-tau_param).*w.*y_in + (z_in==1).*unemp_benefit + transfer ...
    + (transfer*transfer_prop*(H_T(t)==1) - transfer*transfer_prop*(H_T(t)==2)*adjusted_ratio) ...
    - wdl_fix -  transaction_cost_vec) ;  


Bdot = (rb_vec) .* B_vec + (z_in==2).*(1-tau_param).*w.*y_in + (z_in==1).*unemp_benefit + transfer ...
+ (transfer*transfer_prop*(H_T(t)==1) - transfer*transfer_prop*(H_T(t)==2)*adjusted_ratio)  ...
- wdl_fix - consumption_fix - transaction_cost_vec ; 

Adot = (ra_vec).*A_vec + wdl_fix ; 

A_vec_upd = Adot * Delta_sim + A_vec ; 
B_vec_upd = Bdot * Delta_sim + B_vec ;

Adot_T(:,t) = Adot;
Bdot_T(:,t) = Bdot; 


K_T(t)       = mean(A_vec) + mean(B_vec) ;
C_T(t) = mean(consumption_fix) ; 
D_T(t) = mean(wdl_fix) ; 
A_T(t) = mean(A_vec); 
B_T(t) = mean(B_vec) ;
A_vec_t(:,t) = A_vec ;
B_vec_t(:,t) = B_vec ; 
%K_T(t)

A_vec = A_vec_upd; 
B_vec = max(B_vec_upd,b(1)); 
C_vec_t(:,t) = consumption_fix; 
D_vec_t(:,t) = wdl_fix; 

 end

clear all

%% ------------- NEURAL NETWORK PARAMETERS -------------
L = 3; 
N_draw = 5000; S_draw = 200; 
S = S_draw; 
N = N_draw; N_verybig = N_draw; 

phi_numLayers  = 3;
phi_numNeurons = 512;
L = 3;            % 
input_phi = 4;    % e.g number of 
rho_numLayers  = 3; 
rho_numNeurons = 256;
eps_V = 1e-5*0.8; 
eps_CD = 1e-5*0.3 ; 
numEpochs       = 7500;
resampling_time = 2 ; % choose number of resampling 
number_individual_state_variable    = 4 ; 
number_aggregate_state_variable     = 1 ;
initialLearnRate = .001;
 
run('deep_learning_network_V')
run('deep_learning_network_C.m')
run('deep_learning_network_D.m')

parameters_DC.netParamsC = netParamsC; 
parameters_DC.netParamsD = netParamsD; 
%% --------------- CREATE PARAMETERS -------------
dt = 0.025 ; 
run('create_params.m')
Iy = params.Iy2; % number of discrete point in the idionsycratic productivity 
Nz = params.Nz; 
I = 86;
params.I = I ; 
J = params.J_a; 
H = 2; % number of aggregate shock 
%% ---------------- PARAMETERS FOR SIMULATION -----
T_transition = 10000 ;  %point in time dimension 

time = [1:T_transition]*dt  ; 

%% ------------ RUN FUNCTION ---------- 
[gz, g_norm, k_dss , a_dss, b_dss,trans2,gg_t2,b,a,y,mean_z2 ,V_dss, abar, transfer , unemp_benefit, ra, rb_pos] = TwoAsset_Firststep(params) ; 

% ---- Save some parameters that depends on the equilibrium 
params.rb_pos = rb_pos ; 
params.ra = ra; 
params.unemp_benefit = unemp_benefit; 
params.mean_z2 = mean_z2; %save the mean of labour supply, to be used later 
params.abar = abar; 
params.transfer_ss = transfer; 
params.trans2 = trans2; 
params.a = a; 
params.b = b ; 

%------ IDIOSYNCRATIC SHOCKS --------
lam_ue_mid = 1 / params.avg_unemp_duration_mid ; %job finding rate
lam_eu_mid =  (params.target_u_mid/ (1-params.target_u_mid)) * lam_ue_mid  ; %job separation rate 

lam_ue_recession = 1 / params.avg_unemp_duration_recession ; 
lam_eu_recession =  (params.target_u_recession/ (1-params.target_u_recession)) * lam_ue_recession  ;


Y_transition_mid = [-lam_ue_mid , lam_ue_mid ; lam_eu_mid , -lam_eu_mid] ; %more likely U->E in normal time 
Y_transition_recession = [-lam_ue_recession , lam_ue_recession ; lam_eu_recession , -lam_eu_recession] ; 

params.discrete_transition_mid           = cts_to_discrete_prob(Y_transition_mid,dt) ; 
params.discrete_transition_recession     = cts_to_discrete_prob(Y_transition_recession,dt) ; 
params.income_transition_discrete = cts_to_discrete_prob(trans2,dt) ;

 miniBatchSize = 1;

TFP_L = params.z_low;
TFP_M = params.z_mid;
%% Second step: solve many finite agent with agg shock. Assume V_A and V_B = 0 

gz_reshape = reshape(gz,[Iy Nz]) ; 
g_emp = sum(gz_reshape,1) ; 
g_iy2 = sum(gz_reshape,2) ; 

%% ---------- PRESIMULATION OF TFP AND THEN EMP STATE: NEEDED TO DRAW DATA 
% ------------ SIMULATE AGG SHOCK -------------- %% 
[tfp_path_G_long,tfp_path_index_G_long] = discrete_simulation_AggShock_TwoState(params,T_transition,1) ; % just draw one series 

%initialize vector of employment state
z_initial               = [params.target_u_mid, 1-params.target_u_mid]*N_draw ; 
z_index_initial_in      = zeros(N_draw,1); 
z_index_initial_in(1:round(z_initial(1)))      = 1; 
z_index_initial_in(round(z_initial(1))+1:end)  = 2; 
%initialize vector of idiosyncratic state

y_initial = round(gg_t2*N_draw,0) ; 

g_cumsum_idio = [0 ; cumsum(gg_t2)] ;

y_index_initial_in = zeros(N_draw,1); 

for n = 1:N_draw
    u = rand(1,1);
    indx_y = sum(g_cumsum_idio-u<0,1) ;
    y_index_initial_in(n,1) = indx_y;
end

%% ----------- PRESIMULATE EMP_STATE 
% ----------- FIND STEADY STATE FIRST SO LATER WE THROW MASS OF EMP WHEN
% SWITCHING STATE 
y_t_mid = ...
    discrete_simulation_emp_steadystate(z_index_initial_in,params,T_transition,N_verybig,params.discrete_transition_mid) ; 
emp_sss_mid = sum(y_t_mid(:,end-10)==2) ; 
unemp_sss_mid = N_draw - emp_sss_mid ; 

y_t_recession = ...
    discrete_simulation_emp_steadystate(z_index_initial_in,params,T_transition,N_verybig,params.discrete_transition_recession) ; 

emp_sss_recession = sum(y_t_recession(:,end-10)==2) ; 
unemp_sss_recession = N_draw - emp_sss_recession ; 

params.emp_sss_sid = emp_sss_mid; 
params.unemp_sss_mid = unemp_sss_mid; 

params.emp_sss_recession = emp_sss_recession; 
params.unemp_sss_recession = unemp_sss_recession; 


%%  ---------------  CONSTRUCT FOR TRAINING LATER 

temp = [];
az=[];
ij=1;

for h_ind = 1:H
    for o_ind = 1:Nz
        for y_ind = 1:Iy
            for j=1:J
                for i=1:I
                    az=[az,[b(i);a(j);y(y_ind);o_ind;h_ind]]; %NEED TO FIX y1_grid HERE
                end
            end
        end
    end
end

temp = reshape(az,[5 1 I*J*Iy*Nz*H]);
az = dlarray(repmat(temp,[1 miniBatchSize 1]),'CBT');
Prob.az=single(gpuArray(az)) ;


%% ------------------- DATA TO TRAIN NETWORK -------------- 
if resampling_index<=resampling_time %run this until done resampling

if resampling_index == 0  


[z_T,~] = discrete_simulation_emp_2state(z_index_initial_in,tfp_path_index_G_long,params,T_transition,N_verybig) ; 
y_T = discrete_simulation_idiosyncratic(y_index_initial_in,params,T_transition,N_verybig) ;

% -----------------------------
% Burn-in + training window
% -----------------------------
burn_in = 1000;

z_T_train = z_T(:, burn_in+1:T_transition); 
y_T_train = y_T(:, burn_in+1:T_transition); 
G_T_train = tfp_path_index_G_long(burn_in+1:T_transition, 1);   % aggregate state index (T_train x 1)

% -----------------------------
% Target mix: 4/5 good, 1/5 bad
% -----------------------------
S_draw_good = round(S_draw * 4/5);
S_draw_bad  = S_draw - S_draw_good;   % ensures total exactly S_draw

% (Optional safety) make sure there are enough time points in each regime
ind_good = (G_T_train == 2);
ind_bad  = (G_T_train == 1);

z_good = z_T_train(:, ind_good);
y_good = y_T_train(:, ind_good);
G_good = G_T_train(ind_good);

z_bad  = z_T_train(:, ind_bad);
y_bad  = y_T_train(:, ind_bad);
G_bad  = G_T_train(ind_bad);

% -----------------------------
% Preallocate outputs
% -----------------------------
N_draw = size(z_T_train,1);   % number of agents (should equal N_verybig)
data_z_index = zeros(1, N_draw, S_draw);
data_y_index = zeros(1, N_draw, S_draw);
data_h_index = zeros(1, 1, S_draw);

unemp_rate_sind = zeros(1, S_draw);
y_mean          = zeros(1, S_draw);

% -----------------------------
% Sample time indices (without replacement)
% -----------------------------
rand_good = randperm(size(z_good,2), S_draw_good)';
rand_bad  = randperm(size(z_bad,2),  S_draw_bad)';

% -----------------------------
% Fill GOOD draws (first block)
% -----------------------------
for s_ind = 1:S_draw_good
    tt = rand_good(s_ind);

    data_z_index(1,:,s_ind) = z_good(:,tt);
    data_y_index(1,:,s_ind) = y_good(:,tt);
    data_h_index(1,:,s_ind) = G_good(tt);

    unemp_rate_sind(s_ind) = sum(data_z_index(1,:,s_ind) == 1) / N_draw;
end

% -----------------------------
% Fill BAD draws (second block)
% -----------------------------
for s_ind = 1:S_draw_bad
    s_out = S_draw_good + s_ind;  % position in full draw set
    tt = rand_bad(s_ind);

    data_z_index(1,:,s_out) = z_bad(:,tt);
    data_y_index(1,:,s_out) = y_bad(:,tt);
    data_h_index(1,1,s_out) = G_bad(tt);

    unemp_rate_sind(s_out) = sum(data_z_index(1,:,s_out) == 1) / N_draw;
end

data_y = y(data_y_index) ; 

%% Draw conditional distribution: small correction: spread all negative measures around
Iy_sum = params.Iy2*params.Nz; 

gg = zeros(I,J,Iy,Nz) ;
g_norm_test = g_norm; 
measure_neg = abs(sum(g_norm_test(g_norm_test<0),'all')) ; 

[test1,test2] = max(g_norm_test,[],'all') ; 
[indx_b,indx_a,indx_yz] = ind2sub([I J Iy_sum],test2) ; 

g_norm_test(g_norm_test<0) = 0 ; 

g_norm_test(indx_b,indx_a,indx_yz) = g_norm_test(indx_b,indx_a,indx_yz) - measure_neg ; 

g_norm_test = reshape(g_norm_test,[I J Iy Nz]) ; 

for iy2 = 1:Iy
    for nz = 1:Nz
        gg(:,:,iy2,nz) = g_norm_test(:,:,iy2,nz) / sum(g_norm_test(:,:,iy2,nz),'all') ;
    end
end

gg = reshape(gg,[I*J],Iy,Nz) ; 
data_b = zeros(1,N_draw,S_draw); data_a = zeros(1,N_draw,S_draw); 
data_b_index = zeros(1,N_draw,S_draw); data_a_index = zeros(1,N_draw,S_draw); 


for s = 1:S_draw
    for n = 1:N_draw
        cg_ba = [zeros(1,1) ; cumsum(gg(:,data_y_index(1,n,s), data_z_index(1,n,s)),1) ]; %vectorize the joint pdf. recover by ind2sub later
        u = rand(1,1);
        indx_ba = sum(cg_ba-u<0,1) ;
        [indx_b,indx_a] = ind2sub([I J],indx_ba) ;
        data_a_index(1,n,s) = indx_a;
        data_a(1,n,s) = a(indx_a)';
        data_b_index(1,n,s) = indx_b;
        data_b(1,n,s) = b(indx_b);
    end
end

elseif resampling_index > 0 

data_a = zeros(1,N,S) ; data_b = zeros(1,N,S); 
data_a_index = zeros(1,N,S); data_b_index = zeros(1,N,S);
data_z = zeros(1,N,S); data_y = zeros(1,N,S); 
data_z_index = zeros(1,N,S);  data_y_index = zeros(1,N,S); 

data_z(1,:,:)                     = data_z_resample ; 
data_z_index(1,:,:)               = data_z_resample ; 

data_a(1,:,:)                   = data_a_resample ; 
data_a_index(1,:,:)             = data_a_index_resample ; 

data_b(1,:,:)                   = data_b_resample ; 
data_b_index(1,:,:)             = data_b_index_resample ; 

data_y(1,:,:)                     = data_y_resample ; 
data_y_index(1,:,:)               = data_y_index_resample ; 

end


param.y = y; 
params.z = y; 
params.mean_z2 = mean_z2; 

dA_V_DL = zeros(I,J,Iy,Nz,H) ; 
dB_V_DL = zeros(I,J,Iy,Nz,H) ; 

adot_init = zeros(N_draw,H) ; 
bdot_init = zeros(N_draw,H) ; 

Iy2 = params.Iy2; 
V0 = repmat(reshape(V_dss,[I J Iy2 Nz]),[1 1 1 1 H]); 

%% ----------------- SECOND STEP: INITIALIZATION

V_store = zeros(I,J,Iy,Nz,H,S_draw);
c_store = zeros(I,J,Iy,Nz,H,S_draw);
d_store = zeros(I,J,Iy,Nz,H,S_draw);
adot_store = zeros(I,J,Iy,Nz,H,S_draw);
bdot_store = zeros(I,J,Iy,Nz,H,S_draw);

parfor s_ind = 1:S_draw

A_sim = squeeze(data_a(1,:,s_ind))';
B_sim = squeeze(data_b(1,:,s_ind))' ;
A_sim_index = squeeze(data_a_index(1,:,s_ind))';
B_sim_index = squeeze(data_b_index(1,:,s_ind))';
Y_sim_index = squeeze(data_y_index(1,:,s_ind))' ;
Z_sim_index = squeeze(data_z_index(1,:,s_ind))' ;

 [V_temp,c_temp,d_temp,adot_temp,bdot_temp,successful_iteration,flag] = ...
    twoasset_ThirdStep_TwoAggState(V0,params,A_sim,B_sim,...
    A_sim_index,B_sim_index,Y_sim_index,Z_sim_index,0,adot_init,bdot_init,dA_V_DL,dB_V_DL,N_draw,s_ind)  ;

V_store(:,:,:,:,:,s_ind) =  V_temp ;
c_store(:,:,:,:,:,s_ind) =  c_temp ;
d_store(:,:,:,:,:,s_ind) =  d_temp ;
adot_store(:,:,:,:,:,s_ind) =  adot_temp ;
bdot_store(:,:,:,:,:,s_ind) =  bdot_temp ;
end


%% ----- PREPARE OBJECTS FOR DEEP LEARNING: 

Data_a  = single(gpuArray(dlarray(data_a,'CTB')));
Data_z  = single(gpuArray(dlarray(data_z_index,'CTB'))); %z is employment
Data_b  = single(gpuArray(dlarray(data_b,'CTB')));
Data_y = single(gpuArray(dlarray(data_y,'CTB')))     ;  %y is shocks 

%% ------------------------ FIRST STEP: LEARN THIS V IMMEDIATELY: SPLIT INTO TWO NETWORKS. SAME PHI BUT DIFFERENT RHO
V_cell{1} = V_store;
c_cell{1} = c_store;
d_cell{1} = d_store; 
adot_cell{1} = adot_store;
bdot_cell{1} = bdot_store; 

mean_V_A_store_dl_cell{1} = zeros(I,J,Iy,Nz,H,S_draw) ; %initialize to be 0 first
mean_V_B_store_dl_cell{1} = zeros(I,J,Iy,Nz,H,S_draw) ; 

mu_V = mean(vec(V_store)) ;
std_V= sqrt((var(vec(V_store)))) ; 

K_ind = 50; 

for k = 1:K_ind


V_store_DL = V_cell{k};   
V_store_DL_standardize = (V_store_DL - mu_V) / std_V; 

v_store=single(gpuArray(reshape(V_store_DL_standardize,[1 I*J*H*Nz*Iy,S])));
v_store=dlarray(v_store,'CTB');

iteration = 0; 
averageGrad = [];
averageSqGrad = [];
miniBatchSize = 1; 

numEpochs       = 10000;
decayRate = 0.0001;
fun = @error_V; 

accfun = dlaccelerate(fun);
clearCache(accfun)

sample = arrayDatastore([1:S_draw],IterationDimension=2);
mbq = minibatchqueue(sample,MiniBatchSize=miniBatchSize,MiniBatchFormat="BC",OutputEnvironment='gpu');

tic
for epoch = 1:numEpochs
   mbq.shuffle; 
    Loss=0;nmbq=0;LossV=0; LossF= 0;LossE = 0;  Penalty=0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dl = next(mbq);

[gradients,loss,V] = dlfeval(accfun, netParamsV, Data_b(:,dl,:),...
            Data_a(:,dl,:),Data_y(:,dl,:),Data_z(:,dl,:),v_store(:,dl,:),Prob);

        learningRate = initialLearnRate / (1+decayRate*iteration);

        if epoch>1
            [netParamsV,averageGrad,averageSqGrad] = adamupdate(netParamsV,gradients,averageGrad,averageSqGrad,iteration,learningRate);
        end
        Loss = Loss + double(gather(extractdata(loss)));
        nmbq = nmbq+1;
    end
                    initialLearnRate = min(initialLearnRate,10*Loss/nmbq);
                            preallo_loss(epoch) = Loss/nmbq;
    if mod(epoch,1)==0
        disp([Loss/nmbq,learningRate,Loss/nmbq]); 
    end
    if (Loss/nmbq)<(eps_V)
        break
    end
end
toc

epoch_loss(k) = loss;  



%% Get V_A

if epoch > 1

mean_V_A_store_dl_temp = zeros(I,J,Iy,Nz,H,S_draw) ; %V_store_dl = zeros(I,J,s); 
mean_V_B_store_dl_temp = zeros(I,J,Iy,Nz,H,S_draw) ;

%AZ = cat(1,Data_b,Data_a,Data_y,Data_z) ; 
fun = @derivs_V; 

accfun = dlaccelerate(fun); 
clearCache(accfun)

tic
for i = 1:I
    for j = 1:J
        for y_ind = 1:Iy
            for o_ind = 1:Nz
                for h_ind = 1:H
                   % [i,j,y_ind]
                    bsim=b(i);
                    asim=a(j);
                    ysim = y(y_ind) ;
                    azsim = single(gpuArray(dlarray(repmat([bsim;asim;ysim;o_ind;h_ind],[1 s 1]),'CBT'))) ;
                    [V_B,V_A] = dlfeval(accfun,netParamsV,azsim,Data_b,Data_a,Data_y,Data_z);

                    VA_temp = reshape(V_A,[1 1 1 1 1 s N]) ;
                    VB_temp = reshape(V_B,[1 1 1 1 1 s N]);

                    mean_V_A_store_dl_temp(i,j,y_ind,o_ind,h_ind,:) =  mean(extractdata(VA_temp),7) ;
                    mean_V_B_store_dl_temp(i,j,y_ind,o_ind,h_ind,:) =  mean(extractdata(VB_temp),7) ;
                end
            end
        end
    end
end
toc

mean_V_A_store_dl  = std_V*mean_V_A_store_dl_temp ; 
mean_V_B_store_dl  = std_V*mean_V_B_store_dl_temp ; 

end



%% GET c_reverse
mean_V_A_store_dl = reshape(mean_V_A_store_dl,[I J Iy Nz H S_draw]) ; 
mean_V_B_store_dl = reshape(mean_V_B_store_dl,[I J Iy Nz H S_draw]) ; 

c_reverse = zeros(N_draw,H,S_draw);
d_reverse = zeros(N_draw,H,S_draw);
adot_reverse = zeros(N_draw,H,S_draw) ; 
bdot_reverse = zeros(N_draw,H,S_draw) ; 

c_prev = c_cell{k} ; 
d_prev = d_cell{k} ; 
adot_prev = adot_cell{k};   %adot_cell 
bdot_prev = bdot_cell{k};  %bdot_cell


%tic
    for h = 1:H
        for n = 1:N
            for s_ind = 1:S_draw
                c_reverse(n,h,s_ind) = c_prev(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),...
                    data_y_index(1,n,s_ind),data_z_index(1,n,s_ind),h,s_ind) ;

                d_reverse(n,h,s_ind) = d_prev(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),...
                    data_y_index(1,n,s_ind),data_z_index(1,n,s_ind),h,s_ind) ; 

                adot_reverse(n,h,s_ind) = adot_prev(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),...
                    data_y_index(1,n,s_ind),data_z_index(1,n,s_ind),h,s_ind) ;

                bdot_reverse(n,h,s_ind) = bdot_prev(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),...
                    data_y_index(1,n,s_ind),data_z_index(1,n,s_ind),h,s_ind) ;
         
            end
        end
    end

V_store_upd = zeros(I,J,Iy,Nz,H,S_draw) ; 
c_store_upd = zeros(I,J,Iy,Nz,H,S_draw) ; 
d_store_upd = zeros(I,J,Iy,Nz,H,S_draw) ; 
adot_store_upd = zeros(I,J,Iy,Nz,H,S_draw) ; 
bdot_store_upd = zeros(I,J,Iy,Nz,H,S_draw) ; 

Sum_Adot_intial = zeros(S_draw,H); 
Sum_Adot_next = zeros(S_draw,H); 

Sum_Bdot_intial = zeros(S_draw,H); 
Sum_Bdot_next = zeros(S_draw,H);

c_reverse_test = zeros(N,H);
d_reverse_test = zeros(N,H);

V_store = V_cell{k}; 

flag_store_dl = zeros(S_draw,1) ;
T_in = zeros(N_draw,H);
flag_n = zeros(S_draw,1); 


tic
parfor s_ind = 1:S_draw

s_ind
    c_sim       = c_reverse(:,:,s_ind) ; 

    adot_init = adot_reverse(:,:,s_ind) ; 
    bdot_init = bdot_reverse(:,:,s_ind) ; 

    A_sim       = squeeze(data_a(1,:,s_ind))';
    A_sim_index = squeeze(data_a_index(1,:,s_ind))'; 

    B_sim       = squeeze(data_b(1,:,s_ind))';
    B_sim_index = squeeze(data_b_index(1,:,s_ind))'; 

    Z_sim_index = squeeze(data_z_index(1,:,s_ind))'; 
    Y_sim_index = squeeze(data_y_index(1,:,s_ind))'; 

    V_sim = V_store(:,:,:,:,:,s_ind); % to keep iterate on previous 

    if k == 1 
    dA_V_DL = 0.5*mean_V_A_store_dl(:,:,:,:,:,s_ind) ; 
    dB_V_DL = 0.5*mean_V_B_store_dl(:,:,:,:,:,s_ind) ; 
    elseif  k == 2
    dA_V_DL = 0.5*mean_V_A_store_dl(:,:,:,:,:,s_ind) + 0.5*mean_V_A_store_dl_cell{k}(:,:,:,:,:,s_ind) ;
    dB_V_DL = 0.5*mean_V_B_store_dl(:,:,:,:,:,s_ind) + 0.5*mean_V_B_store_dl_cell{k}(:,:,:,:,:,s_ind) ; 
    else
    dA_V_DL = (1/3)*mean_V_A_store_dl(:,:,:,:,:,s_ind) + (1/3)*mean_V_A_store_dl_cell{k}(:,:,:,:,:,s_ind) ...
        + (1/3)*mean_V_A_store_dl_cell{k-1}(:,:,:,:,:,s_ind); 

    dB_V_DL = (1/3)*mean_V_B_store_dl(:,:,:,:,:,s_ind)  + (1/3)*mean_V_B_store_dl_cell{k}(:,:,:,:,:,s_ind) ...
        + (1/3)*mean_V_B_store_dl_cell{k-1}(:,:,:,:,:,s_ind)
    end 

     Sum_Adot_intial(s_ind,:,:) = squeeze(sum(adot_reverse(:,:,s_ind),1)) ;  
     Sum_Bdot_intial(s_ind,:,:) = squeeze(sum(bdot_reverse(:,:,s_ind),1)) ; 

    Adot_prev = Sum_Adot_intial(s_ind,:,:) ; 
    Bdot_prev = Sum_Bdot_intial(s_ind,:,:) ; 


             [V_temp,c_temp,d_temp,adot_temp,bdot_temp,successful_iteration,flag] = ...
    twoasset_ThirdStep_TwoAggState(V_sim,params,A_sim,B_sim,...
    A_sim_index,B_sim_index,Y_sim_index,Z_sim_index,0,adot_init,bdot_init,dA_V_DL,dB_V_DL,N_draw,s_ind)  ;

            flag_n(s_ind) = successful_iteration; 
V_store_upd(:,:,:,:,:,s_ind)    =  V_temp ;
c_store_upd(:,:,:,:,:,s_ind)    =  c_temp ;
d_store_upd(:,:,:,:,:,s_ind)    =  d_temp ;

adot_store_upd(:,:,:,:,:,s_ind) =  adot_temp ;
bdot_store_upd(:,:,:,:,:,s_ind) =  bdot_temp ;
end
toc



 adot_reverse_test = zeros(N_draw,H,S_draw) ; 
 bdot_reverse_test = zeros(N_draw,H,S_draw) ; 

for s_ind = 1:S_draw

for h = 1:H
    for n = 1:N
        adot_reverse_test(n,h,s_ind) =  adot_store_upd(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),data_y_index(1,n,s_ind)...
            ,data_z_index(1,n,s_ind),h,s_ind) ;
        bdot_reverse_test(n,h,s_ind) = bdot_store_upd(data_b_index(1,n,s_ind),data_a_index(1,n,s_ind),data_y_index(1,n,s_ind)...
            ,data_z_index(1,n,s_ind),h,s_ind) ;
    end
end
Sum_Adot_next(s_ind,:,:) = squeeze(sum(adot_reverse_test(:,:,s_ind) )) ;
Sum_Bdot_next(s_ind,:,:) = squeeze(sum(bdot_reverse_test(:,:,s_ind))) ;

end

Atest = [vec(squeeze(Sum_Adot_intial)'),vec(squeeze(Sum_Adot_next)')] ; 
Btest = [vec(squeeze(Sum_Bdot_intial)'),vec(squeeze(Sum_Bdot_next)')] ; 

V_cell{k+1}  =  V_store_upd;% store the first FD one
c_cell{k+1}  =  c_store_upd;
d_cell{k+1}  =  d_store_upd;

adot_cell{k+1}  =  adot_store_upd;
bdot_cell{k+1}  =  bdot_store_upd;

Atest_cell{k} = Atest;
Btest_cell{k} = Btest; 

param_cell{k} = netParamsV;
mean_V_A_store_dl_cell{k+1} = mean_V_A_store_dl;
mean_V_B_store_dl_cell{k+1} = mean_V_B_store_dl;

if k >=3 
    V_cell{k-1} = [];
    c_cell{k-1} = [];
    d_cell{k-1} = [];
end

if k >= 6
    if all(epoch_loss(k-5:k) == 1)
        disp('Convergence achieved: last 5 iterations no update on Value Function.')
        break
    end
end

end

k_end = k; 


%% ---------------         TRAIN CONSUMPTION AND DEPOSIT


C_store_DL =  c_cell{k_end};   
D_store_DL = d_cell{k_end};   

% ---- Consumption 
C_store_DL_standardize_reshape  = reshape(C_store_DL,[I J Iy Nz H S_draw]) ; 
C_store_DL_in                   = single(gpuArray(reshape(C_store_DL_standardize_reshape,[1 I*J*Iy*Nz*H,S_draw])));
C_store_DL_in = dlarray(C_store_DL_in,'CTB');


% ----- Deposit
D_store_DL_standardize_reshape  = reshape(D_store_DL,[I J Iy Nz H S_draw]) ; 
D_store_DL_in   = single(gpuArray(reshape(D_store_DL_standardize_reshape,[1 I*J*Iy*Nz*H,S_draw])));
D_store_DL_in = dlarray(D_store_DL_in,'CTB');

% ----- learning start here 

iteration = 0; 
averageGrad = [];
averageSqGrad = [];
miniBatchSize = 1; 
initialLearnRate = 0.001; 

numEpochs       = 500000;

decayRate = 0.0001;

fun = @error_C_D; 
accfun = dlaccelerate(fun);
clearCache(accfun)

sample = arrayDatastore([1:S_draw],IterationDimension=2);
mbq = minibatchqueue(sample,MiniBatchSize=miniBatchSize,MiniBatchFormat="BC",OutputEnvironment='gpu');

tic
for epoch = 1:numEpochs
    mbq.shuffle;
    Loss=0;nmbq=0;LossV=0; LossF= 0;LossE = 0;  Penalty=0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dl = next(mbq);

        [gradient,loss,C,D,lossC,lossD] = dlfeval(accfun, parameters_DC, Data_b(:,dl,:),...
            Data_a(:,dl,:),Data_y(:,dl,:),Data_z(:,dl,:),...
            C_store_DL_in(:,dl,:),D_store_DL_in(:,dl,:),Prob);

        learningRate = initialLearnRate / (1+decayRate*iteration);

        if epoch>1
            [parameters_DC,averageGrad,averageSqGrad] = adamupdate(parameters_DC,gradient,averageGrad,averageSqGrad,iteration,learningRate);
        end

        Loss = Loss + double(gather(extractdata(loss)));
        nmbq = nmbq+1;
    end
    initialLearnRate = min(initialLearnRate,10*Loss/nmbq);
    preallo_loss(epoch) = Loss/nmbq;
    if mod(epoch,1)==0
          disp([Loss/nmbq,learningRate,lossC/nmbq,lossD/nmbq]);
    end
    if (Loss/nmbq)<(eps_CD)
        break
    end
end
toc


%% ----------------------------- SIMULATION --------------------

%initialize the distribution of A and B
B_in = data_b(1,:,1)' ; 
A_in = data_a(1,:,1)' ; 

[K_T,A_vec_t,B_vec_t,C_vec_t,D_vec_t,Adot_T,Bdot_T] = ...
    TwoAsset_Simulation_Time_TwoAggShock(B_in,A_in,y_T,z_T,...
     tfp_path_index_G_long,params,N_draw,parameters_DC,dt,T_transition) ; 



%% ------------------------------------ RESAMPLING ----------------------------

max_val = T_transition ;
p = 1; % can set to 1 for uniform draw along transitional dynamics 

t = linspace(0,1,S)';
x = 1 + (max_val-1)*t.^p;     % increasing gaps in reals
nums = round(x);              % integer

% force strictly increasing integers
for i = 2:S
    nums(i) = max(nums(i), nums(i-1) + 1);
end

% if we overflow max_val, rescale back to [1,max_val]
if nums(end) > max_val
    nums = round( 1 + (nums - nums(1))*(max_val-1)/(nums(end)-nums(1)) );
    nums(1) = 1;
    for i = 2:S
        nums(i) = max(nums(i), nums(i-1) + 1);
    end
end

data_b_resample               = B_vec_t(:,nums) ; 
data_a_resample               = A_vec_t(:,nums) ; 
data_z_resample               = z_T(:,nums) ; 
data_y_index_resample         = y_T(:,nums) ; 
data_y_resample     = y(data_y_index_resample) ; 

a_grid = a(:);  % ensure column
b_grid = b(:); 
% clip to bounds (so you never map outside grid)
x_a = min(max(data_a_resample, a_grid(1)), a_grid(end));
x_b = min(max(data_b_resample, b_grid(1)), b_grid(end));

% nearest grid index
[~, data_a_index_resample] = min(abs(x_a - reshape(a_grid,1,1,[])), [], 3);
[~, data_b_index_resample] = min(abs(x_b - reshape(b_grid,1,1,[])), [], 3);


% --- increase index 
resampling_index  = resampling_index + 1 ;


%---- clean for memories
K_T = [];
A_vec_t = [];
B_vec_t = []; 
C_vec_t = [];
D_vec_t = [];
Adot_T = []; 
Bdot_T = [] ; 


end



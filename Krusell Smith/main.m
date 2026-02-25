  %% Main
clear all

tic
%%
run('params_construct.m') %params 

I = params.I; 
J = params.J;
H = params.H; 
Z = params.Z; 
T_transitional = 10000;  %length of time for transitional 
dt = 0.01; %time step 
%% DEEP LEARNING RELATED
N_draw = 5000; 
S_draw = 200; 
S = S_draw ; 
N = N_draw; 
phi_numLayers  = 3;
phi_numNeurons = 512;
L = 3;            % 
input_phi = 2;    % e.g., (A,Z): number of distributional state varible 
rho_numLayers  = 3; 
rho_numNeurons = 256;
eps_V = (1e-5)*0.6 ; 
eps_C = (1e-5)*0.3 ; 
numEpochs       = 7500;
resampling_time = 2 ; % choose number of resampling 
phi_VA = 0.5; 
number_individual_state_variable    = 2 ; 
number_aggregate_state_variable     = 1 ;

run('deep_learning_network_C.m') %network for C
run('deep_learning_network_V.m') %network for V 


%% First step: solve continnuum model without aggregate shock 
[V_dss,a,z,c_dss,g,adot,da,r_dss,w_dss,k_dss]  = fd_krusell_smith_continuum_firststep(params) ; 

g_norm = g*da; 
miniBatchSize = 1; 


%% --------------- PRESIMULATE DATA

% ---------------------------- SIMULATE AGGREGATE SHOCK -----------
nval_Zsim = T_transitional;            % ensure consistent length
Zsim   = zeros(nval_Zsim,1);
Zsim(1)= params.Z_mean;

Zshocks = randn(nval_Zsim,1);          % N(0,1) innovations

for t = 2:nval_Zsim
    Zsim(t) = Zsim(t-1) ...
            + params.eta*(params.Z_mean - Zsim(t-1))*dt ...
            + params.sig*sqrt(dt)*Zshocks(t);
end


%% ----------------- DRAW DATA --------------- 


resampling_index = 0 ; %initialize. please don't change it 

if resampling_index<=resampling_time %run this until done resampling

if resampling_index == 0  

g_cond_a = zeros(I,J) ;
for j = 1:J
    g_cond_a(:,j) = g_norm(:,j) / sum(g_norm(:,j)) ; 
end

g_z = sum(g_norm,1); 

% SIMPLER DRAW OF Z: JUST 50/50
z_index = [1*ones(1,N_draw/2),2*ones(1,N_draw/2)]; 
data_z = z(z_index); 

data_a = zeros(1,N_draw,S_draw);
data_a_index = zeros(1,N_draw,S_draw); 

cg_a = [ zeros(1,J) ;cumsum(g_cond_a,1)];

for s = 1:S_draw
    for n = 1:N_draw
        u = rand(1,1);
        indx_a = sum(cg_a(:,z_index(n))-u<0,1) ;
        data_a_index(1,n,s) = indx_a;
        data_a(1,n,s) = a(indx_a);
    end
end

data_z          = repmat(data_z,[1 1 S_draw]); 
data_z_index    = repmat(z_index,[1 1 S_draw]);

elseif resampling_index > 0
% ------------- REASSIGN DATA
data_z = zeros(1,N,S); data_a = zeros(1,N,S) ; data_a_index = zeros(1,N,S);
data_z_index = zeros(1,N,S);

data_z(1,:,:)                   = data_z_resample ; 
data_a(1,:,:)                   = data_a_resample ; 
data_z_index(1,:,:)             = data_z_index_resample ;  
data_a_index(1,:,:)             = data_a_index_resample ; 

end


%% -------------------- PREPARE TRAINING DATA IN DL-CONSISTENT FORMAT ---- 

temp = [];
az=[];
ij=1;
for h = 1:H
    for j=1:J
        for i=1:I
            az=[az,[a(i);z(j);Z(h)]];
        end
    end
end

temp = reshape(az,[3 1 I*J*H]);
az = dlarray(repmat(temp,[1 miniBatchSize 1]),'CBT');
Prob.a=gpuArray(a); Prob.z=gpuArray(z); Prob.az=single(gpuArray(az)) ;

Data_a=single(gpuArray(dlarray(data_a,'CTB')));
Data_z=single(gpuArray(dlarray(data_z,'CTB')));


%% -------------------------------- INITIALIZE VALUE FUNCTION ----------- 
V_store = zeros(I,J,H,S);
c_store = zeros(I,J,H,S);

tic
parfor s = 1:S
    A_sim = squeeze(data_a(1,:,s))';
    Z_sim = squeeze(data_z(1,:,s))' ;
    [V,c] = fd_krusell_smith_finiteagent_secondstep(A_sim,Z_sim,repmat(V_dss,[1 1 H]),params);  %V_FD1
    V_store(:,:,:,s) = V;
    c_store(:,:,:,s) = c;
end
toc


%% ----------------------------  BIG ITERATION

K_iter = 50;
V_cell = cell(K_iter,1);
c_cell = cell(K_iter,1); 
mean_V_A_store_dl_cell = cell(K_iter,1);

V_cell{1} = V_store; 
c_cell{1} = c_store; 
mean_V_A_store_dl_cell{1} = zeros(size(V_store));

std_V = sqrt((var(vec(V_store)))) ;  % to standardize
mean_V = mean(vec(V_store));  % to standardize



 for k = 1:K_iter

V_store_upd = V_cell{k}; 
V_store_upd_std = (V_store_upd - mean_V) / std_V ; 
% Standardization 

averageGrad = [];
averageSqGrad = [];
initialLearnRate = .001;
decayRate = 0.0001;

fun = @error_C; 
accfun = dlaccelerate(fun);
clearCache(accfun)

sample = arrayDatastore([1:S],IterationDimension=2);
mbq = minibatchqueue(sample,MiniBatchSize=miniBatchSize,MiniBatchFormat="BC",OutputEnvironment='gpu');

v_store=single(gpuArray(reshape(V_store_upd_std,[1 I*J*H,S])));
v_store=dlarray(v_store,'CTB');

preallo_loss = zeros(numEpochs,1); 
iteration = 0;

for epoch = 1:numEpochs
   mbq.shuffle; 
    Loss=0;nmbq=0;LossV=0; LossF= 0;LossE = 0;  Penalty=0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dl = next(mbq);
        [gradients,loss,V,penalty,err]  = dlfeval(accfun,netParamsV, Data_a(:,dl,:),Data_z(:,dl,:),v_store(:,dl,:),Prob);
        learningRate = initialLearnRate / (1+decayRate*iteration);

         if epoch>1
        [netParamsV,averageGrad,averageSqGrad] = adamupdate(netParamsV,gradients,averageGrad,averageSqGrad,iteration,learningRate);
         end

        Loss = Loss + double(gather(extractdata(loss)));
        Penalty = max(Penalty,double(gather(extractdata(penalty))));
        nmbq = nmbq+1;
        preallo_loss(epoch) = Loss;
    end
                    initialLearnRate = min(initialLearnRate,5*Loss/nmbq);
    if mod(epoch,10)==0
        disp([Loss/nmbq,Penalty,learningRate]); 
        epoch
    end
    if (Loss/nmbq)<(eps_V)
        break
    end
end

count_epoch_for_convergence(k) = epoch; 



%% ----------------- Get DISTRIBUTIONAL DERIVATIVES ------------

if epoch > 1 %only update if make changes to 

mean_V_A_store_dl = zeros(I,J,H,s); %V_store_dl = zeros(I,J,s); 
AZ = cat(1,Data_a,Data_z); 
fun = @derivs_V; 
accfun = dlaccelerate(fun); 
clearCache(accfun)

for i = 1:I
    for j = 1:J
        for h = 1:H
            asim=a(i);
            zsim=z(j);
            hsim = Z(h);
            azsim = single(gpuArray(dlarray(repmat([asim;zsim;hsim],[1 s 1]),'CBT'))) ;
            V_A = dlfeval(accfun,netParamsV,azsim,AZ);
            V_A = reshape(V_A,[1 1 1 s N]);
            mean_V_A_store_dl(i,j,h,:) = mean(V_A,5) ;
            % V_store_dl(i,j,:) = squeeze(V);
        end
    end
end

 mean_V_A_store_dl =  mean_V_A_store_dl*std_V; 
end


%% ---------------- ITERATE WITH DL V_A 
c_reverse = zeros(N,H,s);
c_prev = c_cell{k};


for h = 1:H
    for n = 1:N
        for s_ind = 1:S_draw
            c_reverse(n,h,s_ind) = c_prev(data_a_index(1,n,s_ind),data_z_index(1,n,s_ind),h,s_ind) ; %Common shocks
        end
    end
end


V_store_upd = zeros(I,J,H,s); c_store_upd = zeros(I,J,H,s); adot_store_upd = zeros(I,J,H,s);
Sum_Adot_intial = zeros(S,H); Sum_Adot_next = zeros(S,H); c_reverse_test = zeros(N,H);
Va_upwind_store_upd = zeros(I,J,H,s); 

V_store = V_cell{k}; 

parfor s_ind = 1:S_draw
    c_sim = c_reverse(:,:,s_ind) ; 
    A_sim = squeeze(data_a(1,:,s_ind))';
    Z_sim = squeeze(data_z(1,:,s_ind))';
    V_sim = V_store(:,:,:,s_ind); % to keep iterate on previous 
    dA_V_DL = (1-phi_VA)*mean_V_A_store_dl(:,:,:,s_ind) + phi_VA*mean_V_A_store_dl_cell{k}(:,:,:,s_ind) ;

    mean_A = mean(A_sim); 
    r =  params.alpha     * exp(Z) * mean_A.^(params.alpha-1)  - params.delta; %interest rates
    w = (1-params.alpha) * exp(Z) * mean_A.^(params.alpha)  ;          %wages
    Sum_Adot_intial(s_ind,:) = sum(A_sim*r + Z_sim*w - c_sim) ;  
  [V,c] = ...
      fd_krusell_smith_finiteagent_thirdstep(A_sim,Z_sim,V_sim,...
      c_sim,dA_V_DL,s_ind,data_a_index,data_z_index,params);

    V_store_upd(:,:,:,s_ind) = V;
    c_store_upd(:,:,:,s_ind) = c;
end


for s_ind = 1:S
    A_sim = squeeze(data_a(1,:,s_ind))';
    Z_sim = squeeze(data_z(1,:,s_ind))';
    mean_A = mean(A_sim);
    r =  params.alpha     * exp(Z) * mean_A.^(params.alpha-1)  -params.delta; %interest rates
    w = (1-params.alpha) * exp(Z) * mean_A.^(params.alpha)  ;          %wages

    c = c_store_upd(:,:,:,s_ind);
     for n = 1:N
        c_reverse_test(n,:) = c(data_a_index(1,n,s_ind),data_z_index(1,n,s_ind),:); %c_store_upd c_FD1_store
     end
    Sum_Adot_next(s_ind,:) = sum(A_sim*r + Z_sim*w -  c_reverse_test) ; 
end

%Atest = [Sum_Adot_intial,Sum_Adot_next]; 
Atest = [vec(Sum_Adot_intial'),vec(Sum_Adot_next')];

V_cell{k+1}  =  V_store_upd;% store the first FD one
c_cell{k+1}  =  c_store_upd;
Atest_cell{k} = Atest;
param_cell{k} = netParamsV;
mean_V_A_store_dl_cell{k+1} = mean_V_A_store_dl;

if k >= 11
    if all(count_epoch_for_convergence(k-11:k) == 1)
        disp('Convergence achieved: last 10 iterations no update on Value Function.')
        break
    end
end

end


k_end = k;  

%% ---------- LEARN CONSUMPTION DECISION  


C_store_upd = c_cell{k_end}; 
averageGrad = [];
averageSqGrad = [];
numEpochs       = 50000;
initialLearnRate = .001;
decayRate = 0.0001;

fun = @error_C; 
accfun = dlaccelerate(fun);
clearCache(accfun)

sample = arrayDatastore([1:S],IterationDimension=2);
mbq = minibatchqueue(sample,MiniBatchSize=miniBatchSize,MiniBatchFormat="BC",OutputEnvironment='gpu');

 std_C = 1 ; 
 mean_C = 0 ; 

C_store_upd_std = (C_store_upd-mean_C)/std_C;

c_store=single(gpuArray(reshape(C_store_upd_std,[1 I*J*H,S])));
c_store=dlarray(c_store,'CTB');

preallo_loss = zeros(numEpochs,1); 
iteration = 0;

tic
for epoch = 1:numEpochs
   mbq.shuffle; 
    Loss=0;nmbq=0;LossV=0; LossF= 0;LossE = 0;  Penalty=0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dl = next(mbq);
        [gradients,loss,V,penalty,err]  = dlfeval(accfun, netParamsC, Data_a(:,dl,:),Data_z(:,dl,:),c_store(:,dl,:),Prob);
        learningRate = initialLearnRate / (1+decayRate*iteration);
        [netParamsC,averageGrad,averageSqGrad] = adamupdate(netParamsC,gradients,averageGrad,averageSqGrad,iteration,learningRate);
        Loss = Loss + double(gather(extractdata(loss)));
        Penalty = max(Penalty,double(gather(extractdata(penalty))));
        nmbq = nmbq+1;
        preallo_loss(epoch) = Loss;
    end
                    initialLearnRate = min(initialLearnRate,5*Loss/nmbq);
    if mod(epoch,50)==0
        disp([Loss/nmbq,Penalty,learningRate]); 
    end
    if (Loss/nmbq)<(eps_C)
        break
    end
end
toc


%% ------------------------------------- SIMULATION ------------------

z0 = data_z(1,:,1) ; 
 
[z_t, Z_t] = simulate_z_forward(z0, T_transitional, dt, params.lam_1, params.y1, params.y2, true) ; 
A_in = data_a(1,:,1)'; 

% ---------------------- SIMULATE BUSINESSS CYCLE ------------------------ 

[K_T,Asim_T_DL] = ...
    krusell_transitional_DL_C(A_in,z_t,Zsim,netParamsC,T_transitional,dt,N,mean_C,std_C) ; 


%% ------------------------------------ RESAMPLING ----------------------------

max_val = T_transitional ;
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


data_a_resample = Asim_T_DL(:,nums) ; 
data_z_resample = z_t(:,nums) ; 
[~, data_z_index_resample] = ismember(data_z_resample, z);

a_grid = a(:);  % ensure column

% clip to bounds (so you never map outside grid)
x = min(max(data_a_resample, a_grid(1)), a_grid(end));

% nearest grid index
[~, data_a_index_resample] = min(abs(x - reshape(a_grid,1,1,[])), [], 3);


% --- increase index 
resampling_index  = resampling_index + 1 ; 

end



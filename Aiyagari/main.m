%% Main
clear all

tic
%%
run('params_construct.m') %params 

I = params.I; 
J = params.J;
T_transitional = 1000;  %length of time for transitional 

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
eps_V = (1e-5)*0.5 ; 
eps_C = (1e-5)*0.3 ; 
numEpochs       = 7500;
resampling_time = 2 ; % choose number of resampling 
phi_VA = 0.5; 
number_individual_state_variable    = 2 ; 
number_aggregate_state_variable     = 0 ;

run('deep_learning_network_C.m') %network for C
run('deep_learning_network_V.m') %network for V 

%% First step: solve continnuum model without aggregate shock 
[V_dss,a,z,c_dss,g,adot,da,r_dss,w_dss,k_dss]  = aiyagari_continuum_firststep(params) ; 
g_norm = g*da;
miniBatchSize = 1; 

%% Draw Training Data from Conditional distribution
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
for j=1:J
    for i=1:I
        az=[az,[a(i);z(j)]];
    end
end

temp = reshape(az,[2 1 I*J]);
az = dlarray(repmat(temp,[1 miniBatchSize 1]),'CBT');
Prob.a=gpuArray(a); Prob.z=gpuArray(z); Prob.az=single(gpuArray(az)) ;

Data_a=single(gpuArray(dlarray(data_a,'CTB')));
Data_z=single(gpuArray(dlarray(data_z,'CTB')));


%% ---------------------- INITIALIZE VALUE FUNCTION V^0 ----------- 
V_store = zeros(I,J);
c_store = zeros(I,J);

tic
for s = 1:S
    A_sim = squeeze(data_a(1,:,s))';
    Z_sim = squeeze(data_z(1,:,s))' ;
    [V,c] = aiyagari_finiteagent_secondstep(A_sim,Z_sim,V_dss,params);  
    V_store(:,:,s) = V;
    c_store(:,:,s) = c;
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
        k
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

v_store=single(gpuArray(reshape(V_store_upd_std,[1 I*J,S])));
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
    if (Loss/nmbq)<((eps_V))
        break
    end
end

count_epoch_for_convergence(k) = epoch; 

%% Get Distributional Derivative 

if epoch > 1 %only update if make changes to 

mean_V_A_store_dl = zeros(I,J,s); 
AZ = cat(1,Data_a,Data_z); 
fun = @derivs_V; 
accfun = dlaccelerate(fun); 
clearCache(accfun)


for i = 1:I
    for j = 1:J
            asim=a(i);
            zsim=z(j);
            azsim = single(gpuArray(dlarray(repmat([asim;zsim],[1 s 1]),'CBT'))) ;
            V_A = dlfeval(accfun,netParamsV,azsim,AZ);
            V_A = reshape(V_A,[1 1 s N]);
            mean_V_A_store_dl(i,j,:) = mean(V_A,4) ; %exploit symmetry to reduce error
        end
end

 mean_V_A_store_dl =  mean_V_A_store_dl*std_V; 
end

%% ---------------- ITERATE WITH DL V_A 
c_reverse = zeros(N,s);
c_prev = c_cell{k};

tic
    for n = 1:N
        for s_ind = 1:s
            c_reverse(n,s_ind) = c_prev(data_a_index(1,n,s_ind),data_z_index(1,n,s_ind),s_ind) ; %Common shocks
        end
    end
toc

V_store_upd = zeros(I,J,s); c_store_upd = zeros(I,J,s); 
Sum_Adot_initial = zeros(S,1); Sum_Adot_next = zeros(S,1); c_reverse_test = zeros(N,1);

V_store = V_cell{k}; 

parfor s_ind = 1:S_draw
    c_sim = c_reverse(:,s_ind) ; 
    A_sim = squeeze(data_a(1,:,s_ind))';
    Z_sim = squeeze(data_z(1,:,s_ind))';
    V_sim = V_store(:,:,s_ind); 
    dA_V_DL = (1-phi_VA)*mean_V_A_store_dl(:,:,s_ind) + phi_VA*mean_V_A_store_dl_cell{k}(:,:,s_ind) ; %weight average between two derivatives to slowly update
    mean_A = mean(A_sim); 
    r =  params.alpha     * mean_A.^(params.alpha-1)  - params.delta; %interest rates
    w = (1-params.alpha) *  mean_A.^(params.alpha)  ;          %wages
    Sum_Adot_initial(s_ind) = sum(A_sim*r + Z_sim*w - c_sim) ;  
  [V,c] = ...
      aiyagari_finiteagent_thirdstep(A_sim,Z_sim,V_sim,c_sim,dA_V_DL,s_ind,data_a_index,data_z_index,params);

    V_store_upd(:,:,s_ind) = V;
    c_store_upd(:,:,s_ind) = c;
end

for s_ind = 1:S
    c_reverse_test = zeros(N,1); 
    for n = 1:N
        c = c_store_upd(:,:,s_ind);
        c_reverse_test(n,:) = c(data_a_index(1,n,s_ind),data_z_index(1,n,s_ind),:); %c_store_upd c_FD1_store
    end
     A_sim = squeeze(data_a(1,:,s_ind))';
    Z_sim = squeeze(data_z(1,:,s_ind))';
    mean_A = mean(A_sim);
    r =  params.alpha     * mean_A.^(params.alpha-1)  -params.delta; %interest rates
    w = (1-params.alpha) *  mean_A.^(params.alpha)  ;          %wages
    Sum_Adot_next(s_ind,1) = sum(A_sim*r + Z_sim*w -  c_reverse_test) ;
end

Atest = [vec(Sum_Adot_initial),vec(Sum_Adot_next)] ;

V_cell{k+1}  =  V_store_upd;% store the first FD one
c_cell{k+1}  =  c_store_upd;
Atest_cell{k} = Atest;
param_cell{k} = netParamsV;
mean_V_A_store_dl_cell{k+1} = mean_V_A_store_dl;

if k >= 21
    if all(count_epoch_for_convergence(k-20:k) == 1)
        disp('Convergence achieved: last 20 iterations no update on Value Function.')
        break
    end
end

    end

k_end = k;  

%% ------------------------------ PLOT EXAMPLE PATH OF ADOT TO EXAMINE CONVERGENCE 
% observe_index = 1 ; 
% Adot_t = zeros(S,k_end); 
% test = Atest_cell{1}; 
% Adot_t(:,1) = test(:,1); 
% for k_test = 2:k_end
%     test = Atest_cell{k_test}; 
%     Adot_t(:,k_test) = test(:,2); 
% end
% 
% figure
% plot([1:k_end],Adot_t(observe_index,:))

%% --------------------------------------- LEARN CONSUMPTION DECISION  
run('deep_learning_network_C.m') %simpler network 

C_store_upd = c_cell{k_end}; 
 std_C = 1; %not standardizing consumption
 mean_C = 0; %not standardizing consumption

C_store_upd_std = (C_store_upd-mean_C)/std_C;

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

c_store=single(gpuArray(reshape(C_store_upd_std,[1 I*J,S])));
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
    if mod(epoch,10)==0
        disp([Loss/nmbq,Penalty,learningRate]); 
    end
    if (Loss/nmbq)<(eps_C)
        break
    end
end
toc


%% ---------------------------------------------- TRANSITIONAL DYNAMICS EXERCISE ---------------------------
%  -------------------- SIMULATE Z_t for some simulation to find S.S
%  consistent with the FD-NN solution 

T = 20000;
Delta_sim = 0.01; 
time = cumsum(Delta_sim*ones(T,1)) ; 
z_t = zeros(N,T);

Z_sim = squeeze(data_z(1,:,1))'; %start from first batch of the training data 
Asim_ss = data_a(1,:,1)'; 

z_t(:,1) = Z_sim; 
eps = rand(N,T); 
Z_t = zeros(T,1); 

prob_switch = round((1-exp(-params.lam_1*Delta_sim))*(N/2)) ; 
low_set = find(z_t(:,1)==params.y1) ;      % low guy
test_low = randsample(low_set,10) ;

high_set = find(z_t(:,1)==params.y2) ; 
test_high = randsample(high_set,10) ; 

z_t(:,2) = z_t(:,1) ;
z_t(test_low,2) = params.y2 ; 
z_t(test_high,2) = params.y1 ; 

for i = 1:T
    z_t(:,i+1) = z_t(:,i) ;
    low_set = find(z_t(:,i)==params.y1) ;      % low guy
    test_low = randsample(low_set,prob_switch) ;
    high_set = find(z_t(:,i)==params.y2) ;
    test_high = randsample(high_set,prob_switch) ;

z_t(test_low,i+1) = params.y2 ; 
z_t(test_high,i+1) = params.y1 ; 

Z_t(i) = mean(z_t(:,i)); 
end


%% ------------------------------------ SIMULATE THE FD-NN SOLUTION SOME PERIODS 
[K_T_DL_ss,Asim_ss_DL] = aiyagari_transitional_DL_C(Asim_ss,z_t,netParamsC,T,Delta_sim,N,mean_C,std_C);
Asim_SS_DL_end = Asim_ss_DL(:,end); 

% ------------------- resimulate z staring from the end of z_t. used to do
% transitional dynamics 
z_start = z_t(:,end); 
Delta_sim = 0.01;
[z_t_cont, Z_t] = simulate_z_forward(z_start, T_transitional, Delta_sim, params.lam_1, params.y1, params.y2, true);


%% ----------------------------------- TRANSITIONAL DYNAMICS: SHIFT THE INITIAL WEALTH DISTRIBUTION 
Asim_ss_DL = Asim_SS_DL_end + 2*da;   %example: injection of wealth 

[K_T_DL_transitional,A_DL_transitional] = ...
    aiyagari_transitional_DL_C(Asim_ss_DL,z_t_cont,netParamsC,T_transitional,Delta_sim,N,mean_C,std_C);



%% ------------------------------------ RESAMPLING ----------------------------

max_val = T_transitional ;
p = 2; % can set to 1 for uniform draw along transitional dynamics 

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


data_a_resample = A_DL_transitional(:,nums) ; 
data_z_resample = z_t_cont(:,nums) ; 
[~, data_z_index_resample] = ismember(data_z_resample, z);

a_grid = a(:);  % ensure column

% clip to bounds (so you never map outside grid)
x = min(max(data_a_resample, a_grid(1)), a_grid(end));

% nearest grid index
[~, data_a_index_resample] = min(abs(x - reshape(a_grid,1,1,[])), [], 3);


% --- increase index 
resampling_index  = resampling_index + 1 ; 
end

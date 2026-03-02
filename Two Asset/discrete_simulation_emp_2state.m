function [z_T,switch_state] = discrete_simulation_emp_2state(z_in,tfp_path_index,param,T,N_verybig)

rng(123) ; 

%factor = N_verybig / 10000; 

%tfp_path_index = tfp_path_index_G(:,1) ; 

%% UNLOAD PARAMETERS 
income_transition_discrete_mid                       = param.discrete_transition_mid  ;  
income_transition_discrete_recession                 = param.discrete_transition_recession ;  
%income_transition_discrete_boom                      = param.discrete_transition_boom      ; 

income_transition_discrete_mid_sss          = income_transition_discrete_mid ; 
income_transition_discrete_recession_sss    = income_transition_discrete_recession ; 
%income_transition_discrete_boom_sss         = income_transition_discrete_boom ; 


for i = 1:10
    income_transition_discrete_mid_sss_upd = income_transition_discrete_mid_sss*income_transition_discrete_mid_sss ; 
    income_transition_discrete_recession_sss_upd = income_transition_discrete_recession_sss*income_transition_discrete_recession_sss ; 
   % income_transition_discrete_boom_sss_upd = income_transition_discrete_boom_sss * income_transition_discrete_boom_sss ; 

   income_transition_discrete_recession_sss = income_transition_discrete_recession_sss_upd ; 
   % income_transition_discrete_boom_sss = income_transition_discrete_boom_sss_upd ; 
   income_transition_discrete_mid_sss = income_transition_discrete_mid_sss_upd; 
end

% income_transition_discrete_mid_sss_upd .* [1,0]'

% emp_LR_sss_recession    = round(income_transition_discrete_recession_sss(1,2)*N_verybig,0) ; 
% unemp_LR_sss_recession  = round(income_transition_discrete_recession_sss(1,1)*N_verybig,0) ; 
% 
% emp_LR_sss_mid    = round(income_transition_discrete_mid_sss(1,2)*N_verybig,0) ; 
% unemp_LR_sss_mid  = round(income_transition_discrete_mid_sss(1,1)*N_verybig,0) ; 
% 
% emp_LR_sss_boom    = round(income_transition_discrete_boom_sss(1,2)*N_verybig,0) ; 
% unemp_LR_sss_boom  = round(income_transition_discrete_boom_sss(1,1)*N_verybig,0) ;

% usual chain: unemp mid = 632 ; unemp recession 781 ; unemp boom 577 

J = length(income_transition_discrete_mid ) ; 
%% --------------- PREPARE SOME OBJECTS 
z_T         = zeros(N_verybig,T) ; 
z_T(:,1)    = z_in;
z_into_loop = z_in; %initial 
z_mat = [1,2] ; 
switch_state = zeros(T,1); 
%% ------------------ INCOME SIMULATION
for t = 2:T

    if tfp_path_index(t,1)-tfp_path_index(t-1,1) == 0 
        switch_state(t,1) = 0 ; 
    else 
        switch_state(t,1) = 1 ;
    end

% switch_state(t,1)
% tfp_path_index(t,1)

    % ---- if mid to recession: throw emp to unemp 
    if switch_state(t,1) == 1 
        z_T(:,t) = z_T(:,t-1); 
        
       if tfp_path_index(t,1) == 1 && tfp_path_index(t-1,1) == 2 % NEED TO THROW E -> U WHEN MID -> RECESSION
           emp_set = find(z_into_loop == 2 )  ; %find emp people 
           current_emp = numel(emp_set) ; %count it 
           low_transit = randsample(emp_set, current_emp - param.emp_sss_recession  ) ; %draw those who need to be unemp
           z_T(low_transit,t) = 1 ; % set them to be emp 

%        elseif tfp_path_index(t,1) == 3 && tfp_path_index(t-1,1) == 2 % NEED TO THROW U -> E WHEN MID -> BOOM
%            unemp_set = find(z_into_loop == 1)  ; %find unemp people 
%            current_unemp = numel(unemp_set) ; %count it 
%            low_transit = randsample(unemp_set, current_unemp - param.unemp_sss_boom ) ; %draw those who need to be emp
%            z_T(low_transit,t) = 2 ; % set them to be emp 

       elseif tfp_path_index(t,1) == 2 && tfp_path_index(t-1,1) == 1 % NEED TO THROW U -> E WHEN RECESSION TO MID 
           unemp_set = find(z_into_loop == 1)  ; %find unemp people 
           current_unemp = numel(unemp_set) ; %count it 
           low_transit = randsample(unemp_set, current_unemp - param.unemp_sss_mid) ; %draw those who need to be emp
           z_T(low_transit,t) = 2 ; % set them to be emp 

%        elseif  tfp_path_index(t,1) == 2 && tfp_path_index(t-1,1) == 3 % NEED TO THROW U -> E WHEN BOOM -> MID       
%            emp_set = find(z_into_loop == 2 )  ; %find emp people 
%            current_emp = numel(emp_set) ; %count it 
%            low_transit = randsample(emp_set, current_emp - param.emp_sss_mid  ) ; %draw those who need to be unemp
%            z_T(low_transit,t) = 1 ; % set them to be emp 
       end
       z_into_loop = z_T(:,t); 


    elseif switch_state(t,1) == 0

       if tfp_path_index(t,1) == 2
            income_transition_discrete = income_transition_discrete_mid ;
        elseif tfp_path_index(t,1) == 1
            income_transition_discrete = income_transition_discrete_recession;
%         elseif tfp_path_index(t,1) == 3
%             income_transition_discrete = income_transition_discrete_boom;
       end

%------ t LOOP. FOR INCOME TRANSITION DISCRERE N 
z_in_next = zeros(N_verybig,1); 
income_transition_discrete_N = zeros(J,J); 
%------- RECONSTRUCT INCOME_DISCRETE EVERYTIME 
   for i = 1:J
        count_z_in(i) = sum(z_into_loop==z_mat(i))  ;
        income_transition_discrete_N(i,:) = income_transition_discrete(i,:)*count_z_in(i)     ;
   end
   
   income_transition_discrete_N = round(income_transition_discrete_N,0) ; %round it so that sampling people
   sum_matrix = sum(income_transition_discrete_N,2) ;  %should sum up to count_z_in, if discrepancy need to fix
    discrepancy_vector = count_z_in' - sum_matrix ;  
    for j = 1:J %throw discrepancy to diagonals 
        income_transition_discrete_N(j,j) = income_transition_discrete_N(j,j) + discrepancy_vector(j) ; 
    end
  %----------- MOVE PEOPLE 
for j = 1:J   % MOVE PEOPLE FROM OTHER STATES TO STATE j 
    low_set = find(z_into_loop == z_mat(j) )  ; %find those who are in state j 
    low_transit = randsample(low_set, sum(income_transition_discrete_N(j,2:end))) ; %sample people who will move; randsample(X,Y): sample Y out of X 
%    low_transit = randsample(low_set, min(numel(low_set),sum(income_transition_discrete_N(j,2:end)))) ; %sample people who will move; randsample(X,Y): sample Y out of X 
%if min hits -> everybody will move 
    low_one = low_set(~ismember(low_set,low_transit)) ; %this is those who end up at state one  
    z_in_next(low_one,1) = z_mat(1) ;

    for i = 2:J-1
        low_set = low_transit ;  % already move those who moved
        low_transit = randsample(low_transit, sum(income_transition_discrete_N(j,i+1:end)))   ; %sample those who move to other states except state i 

      %  low_transit = randsample(low_transit, min(numel(low_transit,sum(income_transition_discrete_N(j,i+1:end)))))   ; %sample those who move to other states except state i 
        low_notone = low_set(~ismember(low_set,low_transit)) ; %this is those who end up at state two
        z_in_next(low_notone,1) = z_mat(i) ;
    end
    z_in_next(low_transit) = z_mat(J) ;
end

z_T(:,t) = z_in_next; 
%mean(z_in_next); 
z_into_loop =  z_in_next ;

%sum(z_T(:,t)==1)/N_draw

    end

end




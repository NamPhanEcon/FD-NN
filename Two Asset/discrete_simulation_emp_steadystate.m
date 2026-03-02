function [z_T] = discrete_simulation_emp_steadystate(z_in,param,T,N_verybig,discrete_idiosyncratic_transition)

z_T         = zeros(N_verybig,T) ; 
z_T(:,1)    = z_in;
z_into_loop = z_in; %initial 
income_transition_discrete = discrete_idiosyncratic_transition ; 

J = length(income_transition_discrete) ; 

z_mat = [1:1:J] ; 

%------ t LOOP. FOR RHO TRANSITION

for t = 2:T

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

end

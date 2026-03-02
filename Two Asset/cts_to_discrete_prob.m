function matrix_outcome = cts_to_discrete_prob(cts_matrix,Delta_sim)

T_trans_temp = cts_matrix; 
T_trans_temp = T_trans_temp  - diag(diag(T_trans_temp)) ;

diag_leave = 1-exp(diag(cts_matrix)*Delta_sim) ;
diag_stay = exp(diag(cts_matrix)*Delta_sim); 
agg_transition_discrete = zeros(size(cts_matrix)); 
T_trans_temp2 = zeros(size(cts_matrix)) ; 

for i = 1:length(agg_transition_discrete)
for j = 1:length(agg_transition_discrete) 
T_trans_temp2(j,i) = (T_trans_temp(j,i) / sum(T_trans_temp(j,:))) *diag_leave(j) ;
end
end

matrix_outcome = T_trans_temp2 + diag(diag_stay) ; % successful 
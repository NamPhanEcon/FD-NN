function [z_T] = discrete_simulation_idiosyncratic(z_in,param,T,N_verybig)

% z_T(n,t) = discrete state of agent n at time t
z_T         = zeros(N_verybig,T);
z_T(:,1)    = z_in;

z_into_loop = z_in; % current states

income_transition_discrete = param.income_transition_discrete; 
J = size(income_transition_discrete,1);

z_mat = (1:J)';

% Preallocate (avoid dynamic resizing inside loops)
count_z_in                 = zeros(J,1);
income_transition_discrete_N = zeros(J,J);
z_in_next                  = zeros(N_verybig,1);

% ------ t LOOP. FOR INCOME TRANSITION DISCRETE
for t = 2:T

    % Reset containers for this period
    z_in_next(:) = 0;
    income_transition_discrete_N(:) = 0;

    % ------- RECONSTRUCT income_transition_discrete_N EVERY TIME
    % Turn probabilities into integer counts (row-by-row) using current population shares
    for i = 1:J
        count_z_in(i) = sum(z_into_loop == z_mat(i));
        income_transition_discrete_N(i,:) = income_transition_discrete(i,:) * count_z_in(i);
    end

    % Round to integers so we can physically "move people"
    income_transition_discrete_N = round(income_transition_discrete_N,0);

    % Fix rounding discrepancy so each row sums to the number of people currently in that state
    sum_matrix = sum(income_transition_discrete_N,2);           % should equal count_z_in
    discrepancy_vector = count_z_in - sum_matrix;               % (+) means missing people; (-) means too many
    for j = 1:J
        income_transition_discrete_N(j,j) = income_transition_discrete_N(j,j) + discrepancy_vector(j);
    end

    % ----------- MOVE PEOPLE
    % For each origin state j, assign agents to destination states 1..J
    for j = 1:J

        low_set = find(z_into_loop == z_mat(j));                % agents currently in origin state j

        % Number who leave origin j (i.e., go to states 2..J); remainder go to state 1
        n_move = sum(income_transition_discrete_N(j,2:end));

        % Sample who moves away from destination 1 (i.e. NOT assigned to state 1)
        if n_move > 0
            low_transit = randsample(low_set, n_move);
        else
            low_transit = [];
        end

        % Those who did NOT get selected to move: assign to destination state 1
        low_one = low_set(~ismember(low_set,low_transit));
        z_in_next(low_one,1) = z_mat(1);

        % Now allocate the moving set across destination states 2..J in sequence
        for i = 2:J-1
            low_set = low_transit;                               % agents not yet assigned
            n_move2 = sum(income_transition_discrete_N(j,i+1:end)); % number who still must be moved further "right"

            if n_move2 > 0
                low_transit = randsample(low_transit, n_move2);
            else
                low_transit = [];
            end

            % Those not selected to move further are assigned to destination state i
            low_notone = low_set(~ismember(low_set,low_transit));
            z_in_next(low_notone,1) = z_mat(i);
        end

        % Whoever remains gets assigned to destination state J
        z_in_next(low_transit,1) = z_mat(J);

    end

    % Save and iterate
    z_T(:,t)      = z_in_next;
    z_into_loop   = z_in_next;

end
function [tfp_path_G, tfp_path_index_G, vec_dist] = discrete_simulation_AggShock_TwoState(param,T,G)
% Simulate exactly G paths of a 2-state aggregate shock with acceptance
% based on time shares close to stationary distribution.

% --- States and transition matrix ---
TFP_mat = [param.z_low, param.z_mid];    % state 1, state 2 values
P = param.agg_transition_discrete;       % 2x2, rows sum to 1

% --- Stationary distribution (your "power to steady state" approach) ---
Psss = param.agg_transition_discrete_sss;
vec_dist = Psss(1,:);                    % stationary dist approx (row vector)

% --- Preallocate outputs ---
tfp_path_G       = zeros(T, G);
tfp_path_index_G = zeros(T, G);

% --- Acceptance settings ---
tol = 0.01;                 % 1% band, same as your 0.99..1.01
max_attempts = 200000;      % guard against infinite loops

g_count = 0;
attempt = 0;

while g_count < G && attempt < max_attempts
    attempt = attempt + 1;

    u_vec = rand(T,1);

    % Initial state
    idx = zeros(T,1);
    idx(1) = 2;

    % Simulate Markov chain
    for t = 2:T
        u = u_vec(t);
        if idx(t-1) == 2
            % from state 2: stay 2 with prob P(2,2), else go to 1
            idx(t) = 2*(u <= P(2,2)) + 1*(u > P(2,2));
        else
            % from state 1: stay 1 with prob P(1,1), else go to 2
            idx(t) = 1*(u <= P(1,1)) + 2*(u > P(1,1));
        end
    end

    % Acceptance test: time shares near stationary distribution
    share1 = mean(idx == 1);
    share2 = 1 - share1;

    if abs(share1 - vec_dist(1)) <= tol*vec_dist(1) && abs(share2 - vec_dist(2)) <= tol*vec_dist(2)
        g_count = g_count + 1;

        tfp_path_index_G(:, g_count) = idx;
        tfp_path_G(:, g_count)       = TFP_mat(idx).';   % map indices -> TFP values
    end
end

if g_count < G
    error('Only accepted %d/%d paths after %d attempts. Loosen tol or increase T / max_attempts.', ...
          g_count, G, attempt);
end
end
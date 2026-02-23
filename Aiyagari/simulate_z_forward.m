function [z_t, Z_t] = simulate_z_forward(z0, T, Delta_sim, lam_1, y1, y2, do_round)
%SIMULATE_Z_FORWARD Simulate cross-sectional two-state process forward.
%
% Inputs
%   z0        : N x 1 initial distribution (values should be y1 or y2)
%   T         : number of time steps to simulate (integer)
%   Delta_sim : time step size (e.g., 0.01)
%   lam_1     : switching intensity
%   y1, y2    : the two state values (e.g., low/high)
%   do_round  : (optional) true -> round number of switchers (your current code)
%                        false -> use probabilistic switching (binomial) [default=false]
%
% Outputs
%   z_t : N x (T+1) matrix, z_t(:,1)=z0 and then forward simulated
%   Z_t : T x 1 vector, mean(z_t(:,t)) for t=1..T

    if nargin < 7 || isempty(do_round)
        do_round = false;
    end

    z0 = z0(:);
    N = length(z0);

    % Basic validation
    if any(z0 ~= y1 & z0 ~= y2)
        error('z0 must contain only y1 and y2 values.');
    end
    if T < 1 || floor(T) ~= T
        error('T must be a positive integer.');
    end

    z_t = zeros(N, T+1);
    Z_t = zeros(T, 1);

    z_t(:,1) = z0;

    % Switching probability per step from Poisson intensity
    p_switch = 1 - exp(-lam_1 * Delta_sim);

    for t = 1:T
        z_t(:,t+1) = z_t(:,t);

        low_set  = find(z_t(:,t) == y1);
        high_set = find(z_t(:,t) == y2);

        % How many to switch from each group this period?
        if do_round
            % Your current "fixed count" approach (same number from low and high)
            k = round(p_switch * (N/2));
            k_low  = min(k, numel(low_set));
            k_high = min(k, numel(high_set));
        else
            % More natural: random number of switchers in each group
            k_low  = min(binornd(numel(low_set),  p_switch), numel(low_set));
            k_high = min(binornd(numel(high_set), p_switch), numel(high_set));
        end

        % Sample who switches (guard for empty sets / zero switches)
        if k_low > 0
            idx_low = randsample(low_set, k_low);
            z_t(idx_low, t+1) = y2;
        end

        if k_high > 0
            idx_high = randsample(high_set, k_high);
            z_t(idx_high, t+1) = y1;
        end

        Z_t(t) = mean(z_t(:,t));
    end
end

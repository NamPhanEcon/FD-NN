params = struct();

% Preferences / household parameters
params.H           = 2; 
params.gamma       = 1.1;
params.load_V      = 1;
params.rb_pos      = 0.02;
params.rb_neg      = 0.065;
params.rho_param   = 0.06175;   % discount rate
params.PerfectAnnuity = 1;
params.tauc     = 5; 
%params.abar = 
% Policy / fiscal parameters
params.abar_frac          = 0.01;
params.unemp_benefit_frac = 0.10;
params.transfer_frac      = 0.05;
params.tau_param          = 0.30;
params.death_prob         = 0;

% Alias (if you need both names)
params.rho = params.rho_param;

% State space / grids
params.z_pos_max = 4;
params.z_neg_min = -0.25;
params.d_step_b  = 0.05;
params.I_b       = 40;
params.J_a       = 40;

params.Icut = params.I_b - 7;
params.zamin = 0;
params.zamax = 11;

% Income process
params.chi0 = 0.25;
params.chi1 = 0.5;
params.chi2 = 1.3350;

params.Iy1 = 3;
params.Iy2 = 5;
params.Iy  = params.Iy1 * params.Iy2;
params.Nz  = 2;

% Solver controls
params.crit  = 1e-5;
params.Delta = 100;
params.maxit = 500;

% Labor market targets
params.avg_unemp_duration = 0.5;
params.target_u = 0.0533;

% Other
params.xi = 0;   % automatic deposit to illiquid

% Production 
params.depreciation_rate       = 0.05 ;  %annual 
params.alpha                   = 0.33;  
params.capital_to_gdp_target   = 3.0 ;
params.delta = params.depreciation_rate; 

%% ------------ BUSINESS CYCLE ------------
%% ===============================
% BUSINESS CYCLE / RECESSION BLOCK
% ===============================

params.expected_recession_duration = 5.5;
params.frequency_recession         = 0.1648;
params.drop_gdp_recession          = 0.9298;

params.target_u_mid       = 0.0533;
params.target_u_recession = 0.0839;

params.avg_unemp_duration_mid       = 2/4;
params.avg_unemp_duration_recession = 3.5/4;

% implied employment rates
params.e_mid       = 1 - params.target_u_mid;
params.e_recession = 1 - params.target_u_recession;

%% ---- TFP calibration ----
lam_TFPshock = 1 / params.expected_recession_duration;

ratio_unemp = (1-params.target_u_recession)^(1-params.alpha) ...
            / (1-params.target_u_mid)^(1-params.alpha);

params.lam_TFPshock = lam_TFPshock;
params.lam_TFP = lam_TFPshock * ...
                 (params.frequency_recession) ...
                 /(1-params.frequency_recession);

params.TFP_M = 1 / ...
    (params.drop_gdp_recession*params.frequency_recession*ratio_unemp ...
     + 1 - params.frequency_recession);

params.TFP_L = ratio_unemp * ...
               params.drop_gdp_recession * params.TFP_M;

params.z_mid = params.TFP_M;
params.z_low = params.TFP_L;

%% ---- Aggregate transition ----
params.agg_transition_small = ...
    [-params.lam_TFPshock ,  params.lam_TFPshock;
      params.lam_TFP      , -params.lam_TFP];

%% ---- Labor-flow implied probabilities ----
params.Q_eu_ML = ...
    (params.e_mid - params.e_recession) / params.e_mid;

params.Q_ue_LM = ...
    (params.target_u_recession - params.target_u_mid) ...
    / params.target_u_recession;

%% ---- Discrete time conversion ------------ 
%----- AGGREGATE SHOCK -----------
TFP_mat = [params.z_low, params.z_mid] ; 

agg_transition_discrete = cts_to_discrete_prob(params.agg_transition_small,dt) ;
agg_transition_discrete_sss = agg_transition_discrete; 

for i = 1:20 %find stationary distribution 
    agg_transition_discrete_sss = agg_transition_discrete_sss*agg_transition_discrete_sss; 
end


params.agg_transition_discrete = agg_transition_discrete;  
params.agg_transition_discrete_sss = agg_transition_discrete_sss ; 


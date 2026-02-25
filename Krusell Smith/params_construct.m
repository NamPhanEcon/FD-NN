%% ------------------ INITIALIZE PARAMETER

params = struct();

params.Z_mean = 0;
params.Z_max  = 0.04;
params.Z_min  = -0.04;

params.eta   = 0.5;     % reversion of TFP shock
params.sig   = 0.01;    % volatility of TFP
params.gamma = 2.1;     % CRRA

params.alpha = 1/3;     % production
params.delta = 0.1;     % depreciation
params.rho   = 0.05;    % discount

params.J     = 2;       % number of idiosyncratic shock points
params.lam_1 = 0.4;
params.lam_2 = 0.4;

params.y1 = 0.3;
params.y2 = 1 + (params.lam_2/params.lam_1) * (1 - params.y1);

params.amin = 1e-6;     % borrowing constraint
params.amax = 25;       % range a
params.I    = 300;      % number of a points
params.H    = 41;       % number of agg shock states

params.Z      = linspace(params.Z_min,params.Z_max,params.H);   % productivity vector

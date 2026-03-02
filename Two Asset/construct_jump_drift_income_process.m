
%% Construct Income Process
sig1 = 1.74; sig2 = 1.53;
factor1 = 1; % one for 
factor2 = 1; 

log_y1_grid = linspace(-sig1*factor1,sig1*factor1,Iy1);
log_y2_grid = linspace(-sig2*factor2,sig2*factor2,Iy2) ; 

dlogy2 = log_y2_grid(end)-log_y2_grid(end-1);
dlogy1 = log_y1_grid(end)-log_y1_grid(end-1);

y1_dist = normpdf(log_y1_grid,0,sig1) ; 
y1_dist = y1_dist / sum(y1_dist) ; 

y2_dist = normpdf(log_y2_grid,0,sig2) ; 
y2_dist = y2_dist / sum(y2_dist) ; 

factor3 = 4; %annualize the quarterly factors reported in paper

lam1    = 0.080*factor3 ;
lam2    = 0.007*factor3;  %jump rate 
beta1   = 0.761*factor3 ;
beta2   = 0.009*factor3;  %decaying rate

%% Transition Matrix for 1: 
trans1_jump = repmat(lam1*y1_dist,[Iy1 1]) - lam1*eye(Iy1,Iy1) ; 
mu_y1 = beta1*(0 - log_y1_grid);        %DRIFT (FROM ITO'S LEMMA)

% Construct decaying of y1
yy      =  - max(mu_y1,0)/dlogy1 + min(mu_y1,0)/dlogy1; %eqn 2  / CENTER
chi     = - min(mu_y1,0)/dlogy1 ; %eqn 1  / CENTER
zeta    =  max(mu_y1,0)/dlogy1 ; %eqn 3 / UPPER

updiag = vec(zeta);
updiag = [0;updiag]; 
centdiag = [ chi(1)+yy(1) ; vec(yy(2:end-1)) ; yy(end)+zeta(end) ]; %reflecting barrier at end point
lowdiag = vec(chi(2:end)); %notice the chi(2:end) here. important 

trans1_drift =  spdiags(centdiag,0,Iy1,Iy1) + spdiags(updiag,1,Iy1,Iy1) + spdiags(lowdiag,-1,Iy1,Iy1) ; %the middle should be 0
trans1 = trans1_drift  + trans1_jump ; %transition for y1 

%% Transition Matrix for 2: 
trans2_jump = repmat(lam2*y2_dist,[Iy2 1]) - lam2*eye(Iy2,Iy2) ; 
mu_y2 = beta2*(0 - log_y2_grid);        %DRIFT (FROM ITO'S LEMMA)
s2_y2 =  0;       %VARIANCE (FROM ITO'S LEMMA)

% Construct decaying of y2
yy      =  - max(mu_y2,0)/dlogy2 + min(mu_y2,0)/dlogy2; %eqn 2  / CENTER
chi     = - min(mu_y2,0)/dlogy2 ; %eqn 1  / CENTER
zeta    =  max(mu_y2,0)/dlogy2 ; %eqn 3 / UPPER

updiag = vec(zeta);
updiag = [0;updiag]; 
centdiag = [ chi(1)+yy(1) ; vec(yy(2:end-1)) ; yy(end)+zeta(end) ]; %reflecting barrier at end point
lowdiag = vec(chi(2:end)); %notice the chi(2:end) here. important 

trans2_drift =  spdiags(centdiag,0,Iy2,Iy2) + spdiags(updiag,1,Iy2,Iy2) + spdiags(lowdiag,-1,Iy2,Iy2) ; %the middle should be 0
trans2 = trans2_drift  + trans2_jump ; %transition for y2

%% --- Sum up two transition into one: just to test the stationary composite distribution 
legal_1 = kron(speye(Iy2),trans1); 
legal_2 = kron(trans2,speye(Iy1));
Trans_all = legal_1 + legal_2;

% Get stationary distribution over z 
T_trans = Trans_all';
Delta_Income = 1000;
gg_t_iy = zeros(Iy,1); gg_t_iy(1) = 1;
for i = 1:1000
          gg_upd = (speye(Iy) - Delta_Income*T_trans) \ gg_t_iy ; % implicit method
gg_t_iy = gg_upd;
end

inc_grid_y1 = exp(log_y1_grid) ; 
inc_grid_y2 = exp(log_y2_grid) ; 


% --- construct income
z_full = zeros(Iy,1);
count = 0; 
for j = 1:Iy2
    for i = 1:Iy1
        count = count+1 ; 
        z_full(count) = exp(log_y1_grid(i))*exp(log_y2_grid(j)); 
    end
end

%% ----------- SEPERATE TRANSITION: 1 first then 2 in the order of state variable

% -------------------- 1 ---------------------
z1  = exp(log_y1_grid) ; 
% Get stationary distribution over z2 
T_trans1 = trans1';
Delta_Income = 1000;
gg_t1 = zeros(Iy1,1); gg_t1(1) = 1;
for i = 1:1000
          gg_upd = (speye(Iy1) - Delta_Income*T_trans1) \ gg_t1 ; % implicit method
gg_t1 = gg_upd;
end
gg_t1 ;

% -------------------- 2 ---------------------
z2  = exp(log_y2_grid) ; 
% Get stationary distribution over z2 
T_trans2 = trans2';
Delta_Income = 1000;
gg_t2 = zeros(Iy2,1); gg_t2(1) = 1;
for i = 1:1000
          gg_upd = (speye(Iy2) - Delta_Income*T_trans2) \ gg_t2 ; % implicit method
gg_t2 = gg_upd;
end
gg_t2 ;

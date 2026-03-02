
function [gz, g_norm,k_dss,a_dss,b_dss,trans2,gg_t2,b,a,z ,mean_z2,V,abar,transfer, unemp_benefit, ra , rb_pos] = ...
    TwoAsset_Firststep(params)


%% --------------- ALL PARAMETERS  ------------------ % 
% ===== Load parameters from struct =====

% Preferences / household parameters
gamma       = params.gamma;
load_V      = params.load_V;
rb_pos      = params.rb_pos;
rb_neg      = params.rb_neg;
rho_param   = params.rho_param;

% Policy / fiscal parameters
abar_frac          = params.abar_frac;
unemp_benefit_frac = params.unemp_benefit_frac;
transfer_frac      = params.transfer_frac;
tau_param          = params.tau_param;

rho = params.rho;

% State space / grids
z_pos_max = params.z_pos_max;
z_neg_min = params.z_neg_min;
d_step_b  = params.d_step_b;
J_a       = params.J_a;

Icut = params.Icut;
zamin = params.zamin;
zamax = params.zamax;

% Income process
chi0 = params.chi0;
chi1 = params.chi1;
chi2 = params.chi2;

Iy1 = params.Iy1;
Iy2 = params.Iy2;
Iy  = params.Iy;
Nz  = params.Nz;

% Solver controls
crit  = params.crit;
Delta = params.Delta;
maxit = params.maxit;

% Labor market targets
avg_unemp_duration = params.avg_unemp_duration;
target_u = params.target_u;
xi = params.xi;

%% --------------- INCOME PROCESS: PERMANENT COMPONENT + UNEMP/EMP TRANSITION -----
run('construct_jump_drift_income_process.m')

lam_ue          = 1 / avg_unemp_duration ; 
lam_eu          = (target_u/ (1-target_u)) * lam_ue  ;
Y_transition    = [-lam_ue , lam_ue ; lam_eu , -lam_eu] ; 
mean_z2         = sum(inc_grid_y2' .* gg_t2) ; 
mean_zfull      = sum(z_full.*gg_t_iy , 'all') ; 

z   = inc_grid_y2' ;  

legal_1_temp = kron(speye(Nz),trans2);  %spread idiosyncratic to emp
legal_2_temp = kron(Y_transition,speye(Iy2)); % spread emp to idiosyncratic prod 

Bswitch_small = legal_1_temp + legal_2_temp;
L = mean_z2 * (1 - target_u) ; 
Iy_sum = Nz*Iy2 ; 

Delta_Income = 100; 
gg_t = zeros(Iy_sum,1); gg_t(1) = 1; 
for i = 1:1000
          gg_upd = (speye(Iy_sum) - Delta_Income*Bswitch_small') \ gg_t ; % implicit method
gg_t = gg_upd;
end


%% Equilibrium 

depreciation_rate       = params.depreciation_rate ;  %annual 
alpha                   = params.alpha;  
capital_to_gdp_target   = params.capital_to_gdp_target;

TFP = 1; 

implied_K = (capital_to_gdp_target)^(1/(1-alpha)) * L ; 
implied_Y = implied_K^(alpha) * L^(1-alpha) ; 
implied_r = TFP*alpha*implied_K^(alpha-1)*L^(1-alpha) - depreciation_rate ; 
implied_w = (1-alpha) * TFP * implied_K^(alpha) * L^(-alpha)   ;          %wages

ra = implied_r; 
w = implied_w ; 

unemp_benefit   = unemp_benefit_frac * implied_Y ; 
transfer        = transfer_frac * implied_Y ; 
abar            = abar_frac*implied_Y; 
%% ------------------------------------------- Grids for b 
b_index = round([z_neg_min:d_step_b:z_pos_max]',2)  ; 
b       = implied_Y * b_index ; 
I       = numel(b) ; 

%% -------------------------------------------  Grids for a 
a_index = linspace(zamin,zamax,J_a)  ; 
a       = implied_Y * a_index ; 
J       = numel(a) ; 
amax    = a(end) ; 

%% -------------------------------------------  Construct step grid
%-- For b
db_f = zeros(I,1); 
db_b = zeros(I,1); 
db_f(1:end-1) = b(2:end) - b(1:end-1);
db_f(end) = db_f(end-1); %will not be used anyway
db_b(2:end) = b(2:end) - b(1:end-1); 
db_b(1) = db_b(2); % will not be used anyway 
db_b = repmat(db_b,[1 J Iy_sum]); 
db_f = repmat(db_f,[1 J Iy_sum]);
bmin = b(1); bmax = b(end); 

%-- For a 
da_f = zeros(1,J); 
da_b = zeros(1,J); 
da_f(1:end-1) = a(2:end) - a(1:end-1);
da_f(end) = da_f(end-1); %will not be used anyway
da_b(2:end) = a(2:end) -a(1:end-1); 
da_b(1) = da_b(2); % will not be used anyway 
da_f = permute(repmat(da_f',[1 I Iy_sum]),[2 1 3]); 
da_b =permute(repmat(da_b',[1 I Iy_sum]),[2 1 3]); 

%% -------------------------------------------  Construct matrix grid 
sss = ones([I J Iy2 Nz]) ;  %1 is unemp
    sss(:,:,:,2) = 2;  %2 is emp
    zzz =  permute(repmat(z,[1 I J Nz]),[2 3 1 4]);
    labour_inc = (1-xi).*w.*zzz*(1-tau_param)  ;  
    inc = labour_inc.*(sss==2) + (sss==1)*unemp_benefit + transfer  ;
    inc = reshape(inc,[I J Iy_sum]) ;


bbb = repmat(b,[1 J Iy_sum]) ; 
aaa = permute(repmat(a',[1 I Iy_sum]),[2 1 3]) ; 

Bswitch = kron(Bswitch_small,speye(I*J))  ; 
ba_0_pos = find(b==0);

%% -------------------------------------------  Natural limit constraint

if b(1)*rb_neg > min(vec(inc))
disp('Natural constraint violated')
end


%% -------------------------------------------  Preallocation
VbF = zeros(I,J,Iy_sum);
VbB = zeros(I,J,Iy_sum);
VaF = zeros(I,J,Iy_sum);
VaB = zeros(I,J,Iy_sum);
c = zeros(I,J,Iy_sum);
updiag = zeros(I*J,Iy_sum);
lowdiag = zeros(I*J,Iy_sum);
centdiag = zeros(I*J,Iy_sum);
AAi = cell(Iy_sum,1);
BBi = cell(Iy_sum,1);


%INITIAL GUESS
v0 = (((1-xi)*inc + ra.*aaa + rb_neg.*bbb).^(1-gamma))/(1-gamma)/rho;
v = v0;
Rb = rb_pos.*(bbb>0) + rb_neg.*(bbb<0);

raa = ra.*ones(1,J);
amax_in = a(end); 
tauc = params.tauc; raa = ra.*(1 - (1.33.*amax_in./a).^(1-tauc)); 

%matrix of illiquid returns
Ra = permute(repmat(raa',[1 I Iy_sum]),[2 1 3]);

M = I*J*Iy_sum; 
           

if load_V == 1 
    load V_temp_quarterly
    v = V;
end

dist = zeros(maxit,1) ; 

for n=1:maxit

    V = v;   
    %DERIVATIVES W.R.T. b
    % forward difference
    VbF(1:I-1,:,:) = (V(2:I,:,:)-V(1:I-1,:,:))./db_f(1:I-1,:,:);
    VbF(I,:,:) = -99999; ((1-xi)*inc(I,:,:) + Rb(I,:,:).*bmax).^(-gamma); %state constraint boundary condition % -99999
    VbB(2:I,:,:) = (V(2:I,:,:)-V(1:I-1,:,:))./db_b(2:I,:,:);
    VbB(1,:,:) = 99999;  ((1-xi)*inc(1,:,:) + Rb(1,:,:).*bmin).^(-gamma); %state constraint boundary condition % 99999

    %I_concave = Vbb > Vbf; 

    %DERIVATIVES W.R.T. a
    % forward difference
    VaF(:,1:J-1,:) = (V(:,2:J,:)-V(:,1:J-1,:))./da_f(:,1:J-1,:);
    % backward difference
    VaB(:,2:J,:) = (V(:,2:J,:)-V(:,1:J-1,:))./da_b(:,2:J,:);
    
    %useful quantities
    c_B = max(VbB,10^(-6)).^(-1/gamma);
    c_F = max(VbF,10^(-6)).^(-1/gamma);
    dBB = two_asset_kinked_FOC_fortran(VaB,VbB,aaa,chi0,chi1,chi2,abar);
    dFB = two_asset_kinked_FOC_fortran(VaB,VbF,aaa,chi0,chi1,chi2,abar);
    %VaF(:,J,:) = VbB(:,J,:).*(1-ra.*chi1 - chi1*w*zzz(:,J,:)./a(:,J,:));
    dBF = two_asset_kinked_FOC_fortran(VaF,VbB,aaa,chi0,chi1,chi2,abar);
    %VaF(:,J,:) = VbF(:,J,:).*(1-ra.*chi1 - chi1*w*zzz(:,J,:)./a(:,J,:));
    dFF = two_asset_kinked_FOC_fortran(VaF,VbF,aaa,chi0,chi1,chi2,abar);
        
    %UPWIND SCHEME
    d_B = (dBF>0).*dBF + (dBB<0).*dBB;
    %state constraints at amin and amax
    d_B(:,1,:) = (dBF(:,1,:)>10^(-12)).*dBF(:,1,:); %make sure d>=0 at amax, don't use VaB(:,1,:)
    d_B(:,J,:) = (dBB(:,J,:)<-10^(-12)).*dBB(:,J,:); %make sure d<=0 at amax, don't use VaF(:,J,:)
    d_B(1,1,:) = max(d_B(1,1,:),0);
    %split drift of b and upwind separately
    sc_B = (1-xi)*inc + Rb.*bbb - c_B;
    sd_B = (-d_B - two_asset_kinked_cost_fortran(d_B,aaa,chi0,chi1,chi2,abar));
    
    d_F = (dFF>0).*dFF + (dFB<0).*dFB;
    %state constraints at amin and amax
    d_F(:,1,:) = (dFF(:,1,:)>10^(-12)).*dFF(:,1,:); %make sure d>=0 at amin, don't use VaB(:,1,:)
    d_F(:,J,:) = (dFB(:,J,:)<-10^(-12)).*dFB(:,J,:); %make sure d<=0 at amax, don't use VaF(:,J,:)
    
    %split drift of b and upwind separately
    sc_F = (1-xi)*inc + Rb.*bbb - c_F;
    sd_F = (-d_F - two_asset_kinked_cost_fortran(d_F,aaa,chi0,chi1,chi2,abar));
    sd_F(I,:,:) = min(sd_F(I,:,:),0);
    
    Ic_B = (sc_B < -10^(-12));
    Ic_F = (sc_F > 10^(-12)).*(1- Ic_B);
    Ic_0 = 1 - Ic_F - Ic_B;
    
    Id_F = (sd_F > 10^(-12));
    Id_B = (sd_B < -10^(-12)).*(1- Id_F);
    Id_B(1,:,:)=0;
    Id_F(I,:,:) = 0; Id_B(I,:,:) = 1; %don't use VbF at bmax so as not to pick up articial state constraint
    Id_0 = 1 - Id_F - Id_B;
    
    c_0 = (1-xi)*inc + Rb.*bbb;
  
    c = c_F.*Ic_F + c_B.*Ic_B + c_0.*Ic_0;
    u = c.^(1-gamma)/(1-gamma);
    
%CONSTRUCT MATRIX BB SUMMARING EVOLUTION OF b
    X = -Ic_B.*sc_B./db_b - Id_B.*sd_B./db_b ;%lower diagonal. impose 0 for all a1 in X 
    XX = X;
    XX(1,:,:) = 0; 
    XX = vec(XX);
    lowdiag = [XX(2:length(XX))];%peculiar of spsdiag
    
    Y = Ic_B.*sc_B./db_b - Ic_F.*sc_F./db_f + Id_B.*sd_B./db_b - (Id_F.*sd_F)./db_f;
    YY = vec(Y); 
    centdiag = YY; 
    
    Z = (Ic_F.*sc_F)./db_f + (Id_F.*sd_F)./db_f; %upp diag
    ZZ = Z;
    ZZ(end,:,:) = 0; 
    ZZ = vec(ZZ); 
    updiag = [0;ZZ];  %peculiar of spsdiag

    BB = spdiags(centdiag,0,M,M) + spdiags(updiag,1,M,M) + spdiags(lowdiag,-1,M,M)  ;
    
    %CONSTRUCT MATRIX AA SUMMARIZING EVOLUTION OF a
    dB = Id_B.*dBB + Id_F.*dFB;
    dF = Id_B.*dBF + Id_F.*dFF;
    MB = min(dB,0);
    MF = max(dF,0) + Ra.*aaa;
    MB(:,J,:) = min(dB(:,J,:) + Ra(:,J,:).*amax,0); %this is hopefully negative
    MF(:,J,:) = 0;

    chi = -MB./da_b;
    yy =  MB./da_b - MF./da_f;
    zeta = MF./da_f;    
    
    chii = chi;
    chii(:,1,:) = 0 ; 
    chii = vec(chii);
    AAlowdiag1 = [chii(I+1:end)]; 
    
AAcentdiag1 = vec(yy); 

zetaa = zeta; 
zetaa(:,end,:) = 0; 
zetaa = vec(zetaa);
zetaa = [zeros(I,1);zetaa];
AAupdiag1 = zetaa; 

AA = spdiags(AAcentdiag1,0,M,M) + spdiags(AAupdiag1,I,M,M) + spdiags(AAlowdiag1,-I,M,M)  ;
    
    A = AA + BB + Bswitch;
    
    if max(abs(sum(A,2)))>10^(-6)
       disp('Improper Transition Matrix')
     % break
    end

   
    B = (1/Delta + rho)*speye(M) - A;
    
    u_stacked = reshape(u,M,1);
    V_stacked = reshape(V,M,1);
    
    vec_temp = u_stacked + V_stacked/Delta;
    
    V_stacked = B\vec_temp; %SOLVE SYSTEM OF EQUATIONS
        
    V = reshape(V_stacked,I,J,Iy_sum);   
    
    
    Vchange = V - v;
    v = V;
    
    dist(n) = max(max(max(abs(Vchange))));
    disp(['Value Function, Iteration ' int2str(n) ', max Vchange = ' num2str(dist(n))]);
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
    
end

d = Id_B.*d_B + Id_F.*d_F;
adot = Ra.*aaa + d ; 
bdot = inc + Rb.*bbb - c - d - two_asset_kinked_cost_fortran(d,aaa,chi0,chi1,chi2,abar) ;

%% ------------ ALTERNATIVE
%CONSTRUCT MATRIX BB SUMMARING EVOLUTION OF b

  X = -min(bdot,0)./db_b;
  Y = min(bdot,0)./db_b - max(bdot,0)./db_f;
  Z = max(bdot,0)./db_f;

  XX = X;
  XX(1,:,:) = 0;
  XX = vec(XX);
  lowdiag = [XX(2:length(XX))];%peculiar of spsdiag

  YY = vec(Y);
  centdiag = YY;

  ZZ = Z;
  ZZ(end,:,:) = 0;
  ZZ = vec(ZZ);
  updiag = [0;ZZ];  %peculiar of spsdiag

    BBtest = spdiags(centdiag,0,M,M) + spdiags(updiag,1,M,M) + spdiags(lowdiag,-1,M,M)  ;
    

% ----------------
chi = -min(adot,0)./da_b ;
yy =  min(adot,0)./da_b - max(adot,0)./da_f;
zeta = max(adot,0)./da_f;
    
chii = chi;
chii(:,1,:) = 0 ; 
chii = vec(chii);
AAlowdiag1_test = [chii(I+1:end)]; 
    
AAcentdiag1_test = vec(yy); 

zetaa = zeta; 
zetaa(:,end,:) = 0; 
zetaa = vec(zetaa);
zetaa = [zeros(I,1);zetaa];
AAupdiag1_test = zetaa; 

AAtest = spdiags(AAcentdiag1_test,0,M,M) + spdiags(AAupdiag1_test,I,M,M) + spdiags(AAlowdiag1_test,-I,M,M)  ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATIONARY DISTRIBUTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = AAtest + BBtest + Bswitch;

AT = A';
% Fix one value so matrix isn't singular:
vectemp = zeros(M,1);
iFix = 1111 ;
vectemp(iFix) =  0.1;
AT(iFix,:) = [zeros(1,iFix-1),1,zeros(1,M-iFix)];


g_stacked = AT\vectemp;
g_sum = g_stacked'*ones(M,1);
g_stacked = g_stacked./g_sum;


%% Prepare objects to get measures

db_f = zeros(I,1); da_f = zeros(1,J); 
db_b = zeros(I,1); da_b = zeros(1,J); 

da_f(1:end-1) = a(2:end) - a(1:end-1);
da_f(end) = da_f(end-1); %will not be used anyway
da_b(2:end) = a(2:end) -a(1:end-1); 
da_b(1) = da_b(2); % will not be used anyway 

db_f(1:end-1) = b(2:end) - b(1:end-1);
db_f(end) = db_f(end-1); %will not be used anyway
db_b(2:end) = b(2:end) -b(1:end-1); 
db_b(1) = db_b(2); % will not be used anyway 

DeltaTilde_b            = zeros(I,1);
DeltaTilde_b(1)         = 0.5*db_f(1);
DeltaTilde_b(2:end-1)   = 0.5*(db_f(2:end-1)+db_b(2:end-1)); %average of two steps
DeltaTilde_b(end)       = 0.5*db_b(end);
DeltaTilde_b_repmat = repmat(DeltaTilde_b,[1 J Iy_sum]); 
grid_DeltaTilde_b = spdiags(vec(DeltaTilde_b_repmat),0,M,M); 

DeltaTilde_a            = zeros(1,J);
DeltaTilde_a(1)         = 0.5*da_f(1);
DeltaTilde_a(2:end-1)   = 0.5*(da_f(2:end-1)+da_b(2:end-1)); %average of two steps
DeltaTilde_a(end)       = 0.5*da_b(end);
DeltaTilde_a_repmat = permute(repmat(DeltaTilde_a',[1 I Iy_sum]),[2 1 3]); 
grid_DeltaTilde_a = spdiags(vec(DeltaTilde_a_repmat),0,M,M);

gg = (grid_DeltaTilde_b.*grid_DeltaTilde_a)\g_stacked;
g = reshape(gg,[I J Iy_sum]);

ga = []; 
%Construct ga
   for j = 1:J
       g_acc = 0; 
       for z_index = 1:Iy_sum
           for i = 1:I
g_acc= g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j) + g_acc; 
           end
       end
ga(j) = g_acc; % ga is integral over [1 3]
   end

%Construct gz
   for z_index = 1:Iy_sum
       g_acc = 0; 
       for j = 1:J
           for i = 1:I
g_acc= g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j) + g_acc; 
           end
       end
gz(z_index) = g_acc; % ga is integral over [1 3]
   end


%Construct g:
gb = []; 
   for i = 1:I
       g_acc = 0;
       for j = 1:J
           for  z_index = 1:Iy_sum
               g_acc= g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j) + g_acc;
           end
       end
       gb(i) = g_acc; % ga is integral over [1 3]
   end

   % Joint distribution
   gb_z = zeros(I,Iy);
   for i = 1:I
       for  z_index = 1:Iy_sum
           g_acc = 0;
           for j = 1:J
               g_acc= g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j) + g_acc;
           end
           gb_z(i,z_index) = g_acc; % ga is integral over [1 3]
       end
   end
gb_z_norm = gb_z ./ sum(gb_z,1); % marginal b|z 
% Joint ab conditional on z: 
gb_a = zeros(I,J);
 for i = 1:I
       for  j = 1:J
           g_acc = 0;
           for z_index = 1:Iy_sum
               g_acc= g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j) + g_acc;
           end
           gb_a(i,j) = g_acc; % ga is integral over [1 3]
       end
 end
 
%Normalize g: 
g_norm = zeros(I,J,Iy_sum); 
  for i = 1:I
       for  z_index = 1:Iy_sum
           for j = 1:J
               g_norm(i,j,z_index) = g(i,j,z_index)*DeltaTilde_b(i)*DeltaTilde_a(j);
           end
       end
  end

% ------------ STATISTICS 

mean_a = sum(g_norm.*aaa,'all') ; 
mean_b = sum(g_norm.*bbb,'all') ; 

k_to_Y = (mean_a+mean_b)/(implied_Y) ;
k_dss = mean_a + mean_b ; 
a_dss = mean_a ; 
b_dss = mean_b ; 
disp(['Capital Demand to GDP: ', num2str(k_to_Y)])
disp(['Target Capital to GDP: ', num2str(3.00)])


%save V_temp_quarterly V
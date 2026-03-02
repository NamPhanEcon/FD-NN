function [V,c,d,adot,bdot,n,flag_nan] = twoasset_ThirdStep_TwoAggState(V0,params,A_vec,B_vec,...
    A_sim_index,B_sim_index,Y_sim_index,Z_sim_index,prop_transfer,adot_init,...
    bdot_init,dA_V_DL,dB_V_DL,N_draw,s_ind) 

%% --------------- ALL PARAMETERS  ------------------ % 
% ===== Load parameters from struct =====

% Preferences / household parameters
flag_nan = 0 ;
implicit  = 2 ;
gamma       = params.gamma;
load_V      = params.load_V;
rb_pos      = params.rb_pos;
rb_neg      = params.rb_neg;
rho_param   = params.rho_param;
tauc = params.tauc; 
abar = params.abar; 

% Policy / fiscal parameters
abar_frac          = params.abar_frac;
unemp_benefit_frac = params.unemp_benefit_frac;
unemp_benefit      = params.unemp_benefit ; 
transfer_frac      = params.transfer_frac;
tau_param          = params.tau_param;
transfer_ss        = params.transfer_ss; 
rho = params.rho;
depreciation_rate = params.depreciation_rate; 

% State space / grids
z_pos_max = params.z_pos_max;
z_neg_min = params.z_neg_min;
d_step_b  = params.d_step_b;
%J_a       = params.J_a;

trans2 = params.trans2; 
H       = params.H; 
I       = params.I; 
J       = params.J_a; 
Iy2     = params.Iy2; 
Icut    = params.Icut;
zamin   = params.zamin;
zamax   = params.zamax;

% Income process
chi0 = params.chi0;
chi1 = params.chi1;
chi2 = params.chi2;

Iy1 = params.Iy1;
Iy2 = params.Iy2;
Iy  = params.Iy2;
Nz  = params.Nz;

% Solver controls
crit  = params.crit;
Delta = 0.8;
maxit = params.maxit;

% Labor market targets
avg_unemp_duration = params.avg_unemp_duration;
target_u = params.target_u;
xi = params.xi;

lam_TFPshock = params.lam_TFPshock ; 
lam_TFP = params.lam_TFP ; 

alpha = params.alpha; 
delta = params.delta; 

mean_z2 = params.mean_z2; 
% ------------- 
Q_ue_LM = params.Q_ue_LM ; 
Q_eu_ML = params.Q_eu_ML ;

% ------ 
a = params.a; 
b = params.b; 
z = params.z; 
%% --------- CONSTRUCT EMPLOYMENT - UNEMPLOYMENT PROCESS 
target_u_mid            = params.target_u_mid; 
target_u_recession      = params.target_u_recession; 

avg_unemp_duration_mid            = params.avg_unemp_duration_mid         ; % in year <-> 2 quarters
avg_unemp_duration_recession      = params.avg_unemp_duration_recession  ; % in year <-> 2.5 quarters

lam_ue_mid = 1 / avg_unemp_duration_mid ; %job finding rate
lam_eu_mid =  (target_u_mid/ (1-target_u_mid)) * lam_ue_mid  ; %job separation rate 

lam_ue_recession = 1 / avg_unemp_duration_recession ; 
lam_eu_recession =  (target_u_recession/ (1-target_u_recession)) * lam_ue_recession  ;

z_low  = params.z_low  ; 
z_mid  = params.z_mid  ; 


Y_transition_mid        = [-lam_ue_mid , lam_ue_mid ; lam_eu_mid , -lam_eu_mid] ;
Y_transition_recession  = [-lam_ue_recession , lam_ue_recession ; lam_eu_recession , -lam_eu_recession] ;

Y_transition = [ Y_transition_recession , zeros(Nz,Nz) ;
    zeros(Nz,Nz) , Y_transition_mid] ; 

Y_transition = sparse(Y_transition); 

Y_transition_big = kron(Y_transition,speye(I*J*Iy2)) ; 

%% ---------- CONSTRUCT AGGREGATE SHOCK PROCESS -------  
agg_transition_med = ...
    [-lam_TFPshock , 0, lam_TFPshock*(1-Q_ue_LM) , lam_TFPshock*Q_ue_LM;
    0 , -lam_TFPshock, 0 , lam_TFPshock  ;
    lam_TFP , 0 , -lam_TFP , 0 ,  ;
    Q_eu_ML*lam_TFP , (1-Q_eu_ML)*lam_TFP , 0 , - lam_TFP];

agg_transition_big = kron(agg_transition_med,speye(I*J*Iy2)) ; 

%% --------------- ALL INCOME PROCESS ------------------

Bswitch_small = kron(speye(Nz*H),trans2);  %spread idiosyncratic to emp
Bswitch_big   = kron(Bswitch_small,speye(I*J)) ; 
L = mean_z2.*[1-target_u_recession,1-target_u_mid] ; % labour supply over aggregate state

%% Equilibrium 

TFP = [z_low,z_mid] ; 

Bmean = mean(B_vec);
Amean = mean(A_vec); 
K =   Bmean + Amean ; 

implied_r = TFP.*alpha.*K.^(alpha-1).*L.^(1-alpha) - depreciation_rate ; 
implied_w = (1-alpha) .* TFP .* K.^(alpha) .* L.^(-alpha) ;          %wages



ra      =  implied_r ;  
rb_pos      = ra - (params.ra - params.rb_pos);
rb_neg      = params.rb_neg; 


w = implied_w; 
w = permute(repmat(w',[1 I J Iy Nz]),[2 3 4 5 1]) ; 

%% -------------------------------------------  Construct step grid
%-- For b
db_f = zeros(I,1); 
db_b = zeros(I,1); 
db_f(1:end-1) = b(2:end) - b(1:end-1);
db_f(end) = db_f(end-1); %will not be used anyway
db_b(2:end) = b(2:end) - b(1:end-1) ; 
db_b(1) = db_b(2); % will not be used anyway 
db_b = repmat(db_b,[1 J Iy Nz H]); 
db_f = repmat(db_f,[1 J Iy Nz H]);
bmin = b(1); bmax = b(end); 

%-- For a 
da_f = zeros(1,J); 
da_b = zeros(1,J); 
da_f(1:end-1) = a(2:end) - a(1:end-1);
da_f(end) = da_f(end-1); %will not be used anyway
da_b(2:end) = a(2:end) -a(1:end-1); 
da_b(1) = da_b(2); % will not be used anyway 
da_f = permute(repmat(da_f',[1 I Iy Nz H]),[2 1 3 4 5]); 
da_b =permute(repmat(da_b',[1 I Iy Nz H]),[2 1 3 4 5]); 
amax = a(end);

%% -------------------------------------------  Construct matrix grid  
transfer = zeros(I,J,Iy,Nz,H) ; 
transfer(:,:,:,:,1) = transfer_ss + transfer_ss*prop_transfer ; 
transfer(:,:,:,:,2) = transfer_ss - transfer_ss*prop_transfer * (params.frequency_recession/(1-params.frequency_recession)); 

bbb = repmat(b,[1 J Iy Nz H]) ; 
aaa = permute(repmat(a',[1 I Iy Nz H]) , [2 1 3 4 5])  ;

sss = ones([I J Iy Nz H]) ;  %1 is unemp 
sss(:,:,:,2,:) = 2;  %2 is emp 

zzz =  permute(repmat(z,[1 I J Nz H]),[2 3 1 4 5]); 

inc =  (1-xi).*w.*zzz.*(sss==2)*(1-tau_param) + (sss==1)*unemp_benefit + transfer ;

%Preallocation
VbF = zeros(I,J,Iy,Nz,H);
VbB = zeros(I,J,Iy,Nz,H);
VaF = zeros(I,J,Iy,Nz,H);
VaB = zeros(I,J,Iy,Nz,H);

%% ------------------ Return at different points in state space
%matrix of liquid returns
rb_pos_mat = permute(repmat(rb_pos',[1 I J Iy Nz]),[2 3 4 5 1]); 
rb_neg_mat = permute(repmat(rb_neg',[1 I J Iy Nz]),[2 3 4 5 1]); 
Rb = rb_pos_mat.*(bbb>=0) + rb_neg_mat.*(bbb<0);

raa = (ra)'.*(1 - (1.33.*amax./a).^(1-tauc))  ; 
Ra = permute(repmat(raa,[1 1 I Iy Nz]),[3 2 4 5 1]) ; 

M = I*J*Nz*H*Iy ; 

maxit_outerloop  = 100; 
sum_Adot = cell(maxit_outerloop ,1) ; 
sum_Bdot = cell(maxit_outerloop ,1) ; 

v =  V0; 

bdot_in = bdot_init;
adot_in = adot_init; 

% --- Precompute linear indices for the simulated states (for fast gathers)
% Dimensions: (I, J, Iy, Nz, H)
% Your indices are: B_sim_index -> i (1..I), A_sim_index -> j (1..J),
%                   Y_sim_index -> iy (1..Iy2), Z_sim_index -> nz (1..Nz)

iB  = B_sim_index(:);
jA  = A_sim_index(:);
iyY = Y_sim_index(:);
nzZ = Z_sim_index(:);

% For each H, build linear indices into (I,J,Iy,Nz,H)
lin_idx = cell(H,1);
for h = 1:H
    lin_idx{h} = sub2ind([I,J,Iy2,Nz,H], iB, jA, iyY, nzZ, h*ones(N_draw,1));
end

for n=1:maxit
    
    Bdot_temp2 = repmat(sum(bdot_in)',[1 I J Iy Nz]) ;
    Bdot = permute(Bdot_temp2,[2 3 4 5 1]) ;
    g_dl_b = dB_V_DL .* Bdot ;

    Adot_temp2 = repmat(sum(adot_in)',[1 I J Iy Nz]) ;
    Adot = permute(Adot_temp2,[2 3 4 5 1]) ;
    g_dl_a = dA_V_DL .* Adot ;

    V = v;   
    %DERIVATIVES W.R.T. b
    % forward difference
    VbF(1:I-1,:,:,:,:) = (V(2:I,:,:,:,:)-V(1:I-1,:,:,:,:))./db_f(1:I-1,:,:,:,:);
    VbF(I,:,:,:,:) =  -99999; %((1-xi)*inc(I,:,:,:,:) + Rb(I,:,:,:,:).*bmax).^(-gamma); %state constraint boundary condition % -99999
    VbB(2:I,:,:,:,:) = (V(2:I,:,:,:,:)-V(1:I-1,:,:,:,:))./db_b(2:I,:,:,:,:);
    VbB(1,:,:,:,:) = 99999 ; %((1-xi)*inc(1,:,:) + Rb(1,:,:).*bmin).^(-gamma); %state constraint boundary condition % 99999

    %DERIVATIVES W.R.T. a
    % forward difference
    VaF(:,1:J-1,:,:,:) = (V(:,2:J,:,:,:)-V(:,1:J-1,:,:,:))./da_f(:,1:J-1,:,:,:);
    % backward difference
    VaB(:,2:J,:,:,:) = (V(:,2:J,:,:,:)-V(:,1:J-1,:,:,:))./da_b(:,2:J,:,:,:);
    
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
    d_B(:,1,:,:,:) = (dBF(:,1,:,:,:)>10^(-12)).*dBF(:,1,:,:,:); %make sure d>=0 at amax, don't use VaB(:,1,:)
    d_B(:,J,:,:,:) = (dBB(:,J,:,:,:)<-10^(-12)).*dBB(:,J,:,:,:); %make sure d<=0 at amax, don't use VaF(:,J,:)
    d_B(1,1,:,:,:) = max(d_B(1,1,:,:,:),0);
    %split drift of b and upwind separately
    sc_B = (1-xi)*inc + Rb.*bbb - c_B;
    sd_B = (-d_B - two_asset_kinked_cost_fortran(d_B,aaa,chi0,chi1,chi2,abar));
    
    d_F = (dFF>0).*dFF + (dFB<0).*dFB;
    %state constraints at amin and amax
    d_F(:,1,:,:,:) = (dFF(:,1,:,:,:)>10^(-12)).*dFF(:,1,:,:,:); %make sure d>=0 at amin, don't use VaB(:,1,:)
    d_F(:,J,:,:,:) = (dFB(:,J,:,:,:)<-10^(-12)).*dFB(:,J,:,:,:); %make sure d<=0 at amax, don't use VaF(:,J,:)
    
    %split drift of b and upwind separately
    sc_F = (1-xi)*inc + Rb.*bbb - c_F;
    sd_F = (-d_F - two_asset_kinked_cost_fortran(d_F,aaa,chi0,chi1,chi2,abar));
    sd_F(I,:,:) = min(sd_F(I,:,:),0);
    
    Ic_B = (sc_B < -10^(-12));
    Ic_F = (sc_F > 10^(-12)).*(1- Ic_B);
    Ic_0 = 1 - Ic_F - Ic_B;
    
    Id_F = (sd_F > 10^(-12));
    Id_B = (sd_B < -10^(-12)).*(1- Id_F);
    Id_B(1,:,:,:,:)=0;
    Id_F(I,:,:,:,:) = 0; Id_B(I,:,:,:,:) = 1; %don't use VbF at bmax so as not to pick up articial state constraint
    Id_0 = 1 - Id_F - Id_B;
    
    c_0 = (1-xi)*inc + Rb.*bbb;
  
    c = c_F.*Ic_F + c_B.*Ic_B + c_0.*Ic_0;
    u = c.^(1-gamma)/(1-gamma);
    
%CONSTRUCT MATRIX BB SUMMARING EVOLUTION OF b
    X = -Ic_B.*sc_B./db_b - Id_B.*sd_B./db_b ;%lower diagonal. impose 0 for all a1 in X 
    XX = X;
    XX(1,:,:,:,:) = 0; 
    XX = XX(:);
    lowdiag = [XX(2:length(XX))];%peculiar of spsdiag
    
    Y = Ic_B.*sc_B./db_b - Ic_F.*sc_F./db_f + Id_B.*sd_B./db_b - (Id_F.*sd_F)./db_f;
    YY = Y(:); 
    centdiag = YY; 
    
    Z = (Ic_F.*sc_F)./db_f + (Id_F.*sd_F)./db_f; %upp diag
    ZZ = Z;
    ZZ(end,:,:,:,:) = 0; 
    ZZ = ZZ(:); 
    updiag = [0;ZZ];  %peculiar of spsdiag

    BB = spdiags(centdiag,0,M,M) + spdiags(updiag,1,M,M) + spdiags(lowdiag,-1,M,M)  ;
    
    %CONSTRUCT MATRIX AA SUMMARIZING EVOLUTION OF a
    dB = Id_B.*dBB + Id_F.*dFB;
    dF = Id_B.*dBF + Id_F.*dFF;
    MB = min(dB,0);
    MF = max(dF,0) + Ra.*aaa;
    MB(:,J,:,:,:) = min(dB(:,J,:,:,:) + Ra(:,J,:,:,:).*amax,0); %this is hopefully negative
    MF(:,J,:,:,:) = 0;

    chi = -MB./da_b;
    yy =  MB./da_b - MF./da_f;
    zeta = MF./da_f;    
    
    chii = chi;
    chii(:,1,:,:,:) = 0 ; 
    chii = vec(chii);
    AAlowdiag1 = [chii(I+1:end)]; 
    
AAcentdiag1 = vec(yy); 

zetaa = zeta; 
zetaa(:,end,:,:,:) = 0; 
zetaa = vec(zetaa);
zetaa = [zeros(I,1);zetaa];
AAupdiag1 = zetaa; 

AA = spdiags(AAcentdiag1,0,M,M) + spdiags(AAupdiag1,I,M,M) + spdiags(AAlowdiag1,-I,M,M)  ;
    
    A = AA + BB + Bswitch_big + agg_transition_big + Y_transition_big ;
    
    if max(abs(sum(A,2)))>10^(-6)
       disp('Improper Transition Matrix')
     % break
    end

%% Implicit

if implicit == 1
    B = (1/Delta + rho)*speye(M) - A;
    
    u_stacked = reshape(u,M,1);
    V_stacked = reshape(V,M,1);
    
    vec_temp = u_stacked + V_stacked/Delta + vec(g_dl_b) + vec(g_dl_a) ;
    
    V_stacked = B\vec_temp; %SOLVE SYSTEM OF EQUATIONS
        
    V = reshape(V_stacked,I,J,Iy,Nz,H);   
    
%% Implicit and explicit
else
B = (1/Delta + rho)*speye(M) - (AA + BB);

u_stacked = reshape(u,M,1);
V_stacked = reshape(V,M,1);
vectemp = u_stacked + ((V_stacked/Delta)+(Bswitch_big + agg_transition_big + Y_transition_big)*V_stacked) ...
    + vec(g_dl_b) + vec(g_dl_a) ;
V_stacked = B\vectemp; %SOLVE SYSTEM OF EQUATIONS
    V = reshape(V_stacked,I,J,Iy,Nz,H);   
end

    Vchange = V - v;
    v = V; 
    dist(n) = max(abs(vec(Vchange))) ;


    if sum(isnan(V)) ~= 0
        disp('Imaginary Value Function')
flag_nan  =  1 ; 
        break
    end

if  mod(n,50)==1
   disp(['Value Function, Iteration ' int2str(n) ', max Vchange = ' num2str(dist(n)) ', Batch = ', num2str(s_ind)]);
end

d = Id_B.*d_B + Id_F.*d_F; 
adot = (Ra).*aaa + d ; 
bdot = inc + (Rb).*bbb - c - d - two_asset_kinked_cost_fortran(d,aaa,chi0,chi1,chi2,abar) ;


    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n) 
        break
    end
   

% --- Vectorized gather
bdot_in_temp = zeros(N_draw,H);
adot_in_temp = zeros(N_draw,H);
for h = 1:H
    bdot_in_temp(:,h) = bdot(lin_idx{h});
    adot_in_temp(:,h) = adot(lin_idx{h});
end
adot_in = adot_in_temp;
bdot_in = bdot_in_temp;


end

end



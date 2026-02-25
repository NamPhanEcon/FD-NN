function [V,c] = fd_krusell_smith_finiteagent_thirdstep(A_sim,Z_sim,V0,c_reverse_in,dA_V_DL,s_ind,data_a_index,data_z_index,params)

%% Parameters
Z_mean = params.Z_mean;
Z_max  = params.Z_max;
Z_min  = params.Z_min;

eta = params.eta;           % reversion of TFP shock
sig = params.sig;           % volatility of TFP

gamma = params.gamma;       % CRRA utility
alpha = params.alpha;       % production
delta = params.delta;       % depreciation
rho   = params.rho;         % discount rate

J     = params.J;           % number of idiosyncratic states
lam_1 = params.lam_1;
lam_2 = params.lam_2;

y1 = params.y1;
y2 = params.y2;

amin = params.amin;         % borrowing constraint
amax = params.amax;         % asset max
I    = params.I;            % number of asset grid points
H    = params.H;            % number of aggregate states

%Z = linspace(params.Z_min, params.Z_max, params.H);
%Z = linspace(Z_min,Z_max,H);   % productivity vector
%Delta = 100; 


%% Construct grid for a and z 
a = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
aa = repmat(a,[1 J H]);

z = [y1,y2]; 
zz = permute(repmat(z',[1 I H]),[2 1 3]) ; 

%% Construct transition matrix for idiosyncratic shocks : [y1 y2]
y_transition_small = [-lam_1 , lam_1 ; lam_2 , -lam_2];
y_transition_big   = kron(y_transition_small,speye(I,I)); 
y_transition_big   = kron(speye(H,H),y_transition_big); 

%% Construct transition matrix for aggregate shocks
Z = linspace(Z_min,Z_max,H);   % productivity vector


dZ = (Z_max-Z_min)/(H-1) ;
dZ2 = dZ^2;
ZZ = permute(repmat(Z',[1 I J]),[2 3 1]) ; 
s2 = sig^2; 
mu = eta*(Z_mean - Z);        %DRIFT (FROM ITO'S LEMMA)

yy = - s2/dZ2 - max(mu,0)/dZ + min(mu,0)/dZ ; %eqn 2
chi =  s2/(2*dZ2) - min(mu,0)/dZ; %eqn 1   %SOMEHOW IGNORE THE MU- 
zeta = max(mu,0)/dZ + s2/(2*dZ2); %eqn 3

updiag = vec(zeta);
updiag = [0;updiag]; 
centdiag = [ chi(1)+yy(1) ; vec(yy(2:end-1)) ; yy(end)+zeta(end) ]; %reflecting barrier at end point
lowdiag = vec(chi(2:end)); %notice the chi(2:end) here. important 

Z_transition_small =  spdiags(centdiag,0,H,H) + spdiags(updiag,1,H,H) + spdiags(lowdiag,-1,H,H) ; %the middle should be 0
Z_transition_big   = kron(Z_transition_small,eye(I*J,I*J)); 

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J,H);             
Vab = zeros(I,J,H);
Vzf = zeros(I,J,H);
Vzb = zeros(I,J,H);
Vzz = zeros(I,J,H);
c = zeros(I,J,H);

crit = 1e-6; 
maxit= 10000; 
Delta = 0.90; 

K = mean(A_sim);      % initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
N_agent = numel(A_sim); 
%----------------------------------------------------
%INITIAL GUESS
r =  alpha  * exp(ZZ) * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * exp(ZZ) * K^(alpha);          %wages
% v0 = (w.*zz + r.*aa).^(1-gamma)/(1-gamma)/rho;
% v = v0 ;

r_sim = alpha  * exp(Z) * K^(alpha-1) -delta; %interest rates
w_sim = (1-alpha) * exp(Z) * K^(alpha);          %wages
%v = repmat(V0,[1 1 H]); 
dist = zeros(1,maxit);
% Initialization
c_sim  =  c_reverse_in; %initilization
v = V0;


% Inner Loop 
   for n=1:maxit
       g_dl = dA_V_DL .* reshape(repmat(sum(A_sim*r_sim + Z_sim*w_sim - c_sim),[I*J 1]),[I J H]);
        V = v;
        % forward difference
        Vaf(1:I-1,:,:) = (V(2:I,:,:)-V(1:I-1,:,:))/da;
        Vaf(I,:,:) = (w(I,:,:).*zz(I,:,:) + r(I,:,:).*amax).^(-gamma); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:,:) = (V(2:I,:,:)-V(1:I-1,:,:))/da;
        Vab(1,:,:) = (w(1,:,:).*zz(1,:,:) + r(1,:,:).*amin).^(-gamma);  %state constraint boundary condition

        I_concave = Vab > Vaf;              %indicator whether value function is concave (problems arise if this is not the case)

        %consumption and savings with forward difference
        cf = max(Vaf,10e-6).^(-1/gamma);
        sf = w.*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = max(Vab,10e-6).^(-1/gamma);
        sb = w.*zz + r.*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w.*zz + r.*aa;
        Va0 = c0.^(-gamma);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = sf > 0; %positive drift --> forward difference
        Ib = sb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
   
        Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term

        c = Va_Upwind.^(-1/gamma);
        u = c.^(1-gamma)/(1-gamma);

        %CONSTRUCT MATRIX A
        X = - min(sb,0)/da;
        Y = - max(sf,0)/da + min(sb,0)/da;
        Ztemp = max(sf,0)/da;
        
        X(1,:,:)    = 0;
        Ztemp(end,:,:)  = 0; 

        centdiag = vec(Y);
        updiag = vec(Ztemp);
        lowdiag = vec(X(2:end)); 
        
        AA=spdiags(centdiag,0,I*J*H,I*J*H)+spdiags([0;updiag],1,I*J*H,I*J*H)+spdiags(lowdiag,-1,I*J*H,I*J*H);
        
        A = AA + y_transition_big + Z_transition_big;
        B = (1/Delta + rho)*speye(I*J*H) - A;

        u_stacked = reshape(u,I*J*H,1);
        V_stacked = reshape(V,I*J*H,1);

        b = u_stacked + V_stacked/Delta + vec(g_dl);

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J,H);

        Vchange = V - v;
        v = V;
      %  max(max(max(abs(Vchange))))
        dist(n) = max(max(max(abs(Vchange))));
     %   dist(n)
        if dist(n)<crit
            disp('Value Function Converged, Iteration = ')
            disp(n)
            break
        end
        
        % if (n>1)&(dist(n)<dist(n-1))
        %     Delta = Delta*0.95;
        % end

        for n_agent = 1:N_agent
            c_reverse_test(n_agent,:) = c(data_a_index(1,n_agent,s_ind),data_z_index(1,n_agent,s_ind),:) ; %c_store_upd c_FD1_store
        end

    c_sim = c_reverse_test ; 

   end

end
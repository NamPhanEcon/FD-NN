 function [V,c,Va_Upwind] = aiyagari_finiteagent_secondstep(A_sim,Z_sim,V0,params)

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

Z = linspace(params.Z_min, params.Z_max, params.H);
Z = linspace(Z_min,Z_max,H);   % productivity vector

Delta = 100; 
%% Construct grid for a and z 
a = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
aa = repmat(a,[1 J]);

z = [y1,y2]; 
zz = permute(repmat(z',[1 I]),[2 1]) ; 

%% Construct transition matrix for idiosyncratic shocks : [y1 y2]
y_transition_small = [-lam_1 , lam_1 ; lam_2 , -lam_2];
y_transition_big   = kron(y_transition_small,speye(I,I)); 

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

crit = 1e-6; 
maxit= 1000; 
Delta = 100; 

K = mean(A_sim);      % initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
%----------------------------------------------------
%INITIAL GUESS
r =  alpha  * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * K^(alpha);          %wages
v0 = (w.*zz + r.*aa).^(1-gamma)/(1-gamma)/rho;
v = V0 ;
dist = zeros(1,maxit);

% Inner Loop 
   for n=1:maxit
        V = v;
        % forward difference
        Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vaf(I,:) = (w.*zz(I,:) + r.*amax).^(-gamma); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vab(1,:) = (w.*zz(1,:) + r.*amin).^(-gamma);  %state constraint boundary condition

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
        
        X(1,:)    = 0;
        Ztemp(end,:)  = 0; 

        centdiag = vec(Y);
        updiag = vec(Ztemp);
        lowdiag = vec(X(2:end)); 
        
        AA=spdiags(centdiag,0,I*J,I*J)+spdiags([0;updiag],1,I*J,I*J)+spdiags(lowdiag,-1,I*J,I*J);
        
        A = AA + y_transition_big;
        B = (1/Delta + rho)*speye(I*J) - A;

        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);

        b = u_stacked + V_stacked/Delta;

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J);

        Vchange = V - v;
        v = V;

        dist(n) = max(max(max(abs(Vchange))));
        if dist(n)<crit
            disp('Value Function Converged, Iteration = ')
            disp(n)
            break
        end

   end

end
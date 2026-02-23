function [V,a,z,c,g,adot,da,r,w,K,Va_Upwind] = aiyagari_continuum_firststep(params)

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

%% Construct grid for a and z 
a = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
z = [y1,y2]; 
aa = a*ones(1,J);
zz = ones(I,1)*z;

%% Construct transition matrix: [y1 y2]
y_transition_small = [-lam_1 , lam_1 ; lam_2 , -lam_2];
y_transition_big   = kron(y_transition_small,eye(I,I)); 

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

critK = 1e-3;
crit = 1e-8; 
maxitK = 400; 
maxit= 1000; 
K = 4.91;      % initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
relax = 0.99; % relaxation parameter 
Delta = 100; 
%----------------------------------------------------
%INITIAL GUESS
r =  alpha     * exp(Z_mean) * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * exp(Z_mean) * K^(alpha);          %wages
v0 = (w*zz + r.*aa).^(1-gamma)/(1-gamma)/rho;
v0 = zeros(I,J); 
v = v0;
dist = zeros(1,maxit);

%-----------------------------------------------------
% Outer Loop 
for iter=1:maxitK
% Inner Loop 
   for n=1:maxit
        V = v;
        % forward difference
        Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vaf(I,:) = (w*z + r.*amax).^(-gamma); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vab(1,:) = (w*z + r.*amin).^(-gamma);  %state constraint boundary condition

        I_concave = Vab > Vaf;              %indicator whether value function is concave (problems arise if this is not the case)

        %consumption and savings with forward difference
        cf = max(Vaf,10e-6).^(-1/gamma);
        sf = w*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = max(Vab,10e-6).^(-1/gamma);
        sb = w*zz + r.*aa - cb;
        %consumption and derivative of value function at steady state
        c0 = w*zz + r.*aa;
        Va0 = c0.^(-gamma);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = sf > 0; %positive drift --> forward difference
        Ib = sb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        %make sure backward difference is used at amax
        %     Ib(I,:) = 1; If(I,:) = 0;
        %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS sf > 0:
        %already taken care of automatically

        Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term

        c = Va_Upwind.^(-1/gamma);
        u = c.^(1-gamma)/(1-gamma);

        %CONSTRUCT MATRIX A
        X = - min(sb,0)/da;
        Y = - max(sf,0)/da + min(sb,0)/da;
        Z = max(sf,0)/da;
        
        yvec = vec(Y); 
        xvec = vec(X); 
        zvec =vec(Z); 
        sbvec = vec(sb); 
        
        updiag=[0]; %This is needed because of the peculiarity of spdiags.
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag=reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        AA=spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
        
        A = AA + y_transition_big;
        B = (1/Delta + rho)*speye(I*J) - A;

        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);

        b = u_stacked + V_stacked/Delta;

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J);

        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        if dist(n)<crit
            disp('Value Function Converged, Iteration = ')
            disp(n)
            break
        end
   end

    % FOKKER-PLANCK EQUATION %
    AT = A';
    b = zeros(I*J,1);

    %need to fix one value, otherwise matrix is singular
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,I*J-i_fix)];
    AT(i_fix,:) = row;

    %Solve linear system
    gg = AT\b;
    g_sum = gg'*ones(I*J,1)*da;
    gg = gg./g_sum;

    g = reshape(gg,I,J);
    
    % Update aggregate capital
    S = sum(g'*a*da);
   
    clear A AA AT B
    if abs(K-S)<critK
        disp('Equilibrium Found')
        break
    end
    
    %update prices
    K = relax*K +(1-relax)*S;           %relaxation algorithm (to ensure convergence)
    r = alpha     * exp(Z_mean) *  K^(alpha-1) -delta; %interest rates
    w = (1-alpha) * exp(Z_mean)* K^(alpha);          %wages
 
end


 adot = w*zz + r*aa - c; 
%  sum(g,'all')*da
%  sum(adot.*g,'all')*da
% sum(c.*g,'all')*da*dz

end
function [gradient,loss,C,D,lossC,lossD] = ...
    error_C_D(parameters,B,A,Y,Z,Ctar,Dtar,Prob) 

% sum over the first four dimensions

baz = Prob.az;
BAZ = [B;A;Y;Z] ;

C = model_DL_C(parameters.netParamsC,baz,BAZ);
D = model_DL_C(parameters.netParamsD,baz,BAZ);

errC = dlarray(C(:) - Ctar(:),'B');
errD = dlarray(D(:) - Dtar(:),'B');

zerotarget = zeros(size(errC),'like',errC);

%penalty = 0.0001*max(abs(err)); % 0.001*max(abs(err))
%loss = mse(err,zerotarget) + penalty;

lossC = mse(errC,zerotarget); 
lossD = mse(errD,zerotarget) ;

loss = lossC + lossD; 
gradient = dlgradient(loss,parameters);
end
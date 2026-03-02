function [gradient,loss,V] = error_V(parameters,B,A,Y,Z,Vtar,Prob)

% sum over the first four dimensions
baz = Prob.az;
BAZ = [B;A;Y;Z] ;

V = model_DL_C(parameters,baz,BAZ); 

err = dlarray(V(:) - Vtar(:),'B');
zerotarget = zeros(size(err),'like',err);
loss = mse(err,zerotarget);

gradient = dlgradient(loss,parameters);
end
function [gradient,loss,V,penalty,err] = error_C(parameters,A,Z,Vtar,Prob)

az = Prob.az;
AZ = [A;Z];
V = model_DL_C(parameters,az,AZ);

err = dlarray(V(:) - Vtar(:),'B');
zerotarget = zeros(size(err),'like',err);
penalty = .001*max(abs(err));
loss = mse(err,zerotarget) + penalty;

gradient = dlgradient(loss,parameters);
end
function parameter = initializeNormalizedXavier(sz)


%sz = [output input]
lower = -(sqrt(6)/sqrt(sum(sz))) ;
upper = (sqrt(6)/sqrt(sum(sz)));

parameter = lower + rand(sz) * (upper - lower);
parameter = dlarray(parameter);

end


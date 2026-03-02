function eq = two_asset_kinked_cost_fortran(d,a,chi0,chi1,chi2,abar)
% global chi0 chi1 chi2 abar

dtilde = d./max(a,abar); 
cost_tilde = chi0*abs(dtilde) + abs(dtilde).^(1+chi2)*chi1^(-chi2)*(1/(1+chi2));
eq = cost_tilde.*max(a,abar); 

end
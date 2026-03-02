function d = two_asset_kinked_FOC_fortran(pa,pb,a,chi0,chi1,chi2,abar)
%global chi0 chi1 chi2 abar phi_d tau d_bar

LHS =  chi1*(pa./pb - 1 - chi0);
LHS_d = (LHS>0).*LHS.^(1/chi2); 

RHS =  chi1*(pa./pb - 1  + chi0);
RHS_d =  -(RHS<0).*(abs(RHS).^(1/(chi2))); 

d = LHS_d.*max(a,abar) + RHS_d.*a ; 

end



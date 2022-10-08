
function w = polycurvefit_reg(x,t,M,lamda)
  [N,m]= size(x);
  A = zeros(M+1,M+1);
  C =zeros(M+1,1);
  for j =0:M;
    cj =0;
    for i = 0:M;
      aij =0;
      for n =1:N;
        aij += x(n,1)^(i+j);
      if(i==j);
        A(j+1,i+1) = aij +lamda;
      else;
        A(j+1,i+1) = aij;
      endif;
      endfor;
    endfor;
    for n =1:N;
      cj += t(n,1)*(x(n,1)^j);
    endfor;
    C(j+1,1) =cj;
  endfor;
#  w =linsolve (A, C);
  w = inv(A)*C;
endfunction;

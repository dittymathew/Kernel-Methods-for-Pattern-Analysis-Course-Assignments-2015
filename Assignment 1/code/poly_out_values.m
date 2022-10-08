function Y = poly_out_values(w,x)
  [M,m]  =size(w);
  [N,m] =size(x);
  Y= zeros(N,1);
  for n =1:N;
    y_xn =w(1,1);
    for i =2:M;
      y_xn += w(i,1)*(x(n,1)^(i-1));
    endfor;
    Y(n,1)=y_xn;
  endfor;
endfunction

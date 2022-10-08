function error =rsserror(y_predict,y);
  error = sqrt(mean((y_predict-y).^2));
endfunction

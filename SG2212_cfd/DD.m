function A = DD(n,h)
% DD(n,h)
%
% One-dimensional finite-difference derivative matrix 
% of size n times n for second derivative:
%
% This function belongs to SG2212.m
v = -2.*ones(1,n);
A= zeros(n,n);
A = diag(v); 
for j = 1:n 
    if j ==1 
        A(1,j) = -1;
        A(1,j+1) = 1;
    elseif j == n 
        A(end,j) = -1;
        A(end,j-1) = 1;
    else
        A(j,j-1) = 1; A(j,j+1) = 1; 
    end
end
A = (1/(h^2)).*A;
end


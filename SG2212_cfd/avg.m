function B = avg(A,idim)
% AVG(A,idim)
%
% Averaging function to go from cell centres (pressure nodes)
% to cell corners (velocity nodes) and vice versa.
% avg acts on index idim; default is idim=1.
%
% This function belongs to SG2212.m

if nargin<2, idim = 1; end

if (idim==1)
  B = (A( 2:end ,: )+A( 1:end-1 , : ))/2; 
elseif (idim==2)
  B = (A( : , 2:end )+A( : , 1:end-1 ))/2; 
else
  error('avg(A,idim): idim must be 1 or 2')
end
end
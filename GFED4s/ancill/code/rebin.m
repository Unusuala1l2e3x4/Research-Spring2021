function b = rebin (a, m, dim)
% rebin array 'a' along dimension number 'dim' to 'm' elements
% conserves sum over all elements
%
s = size(a);
if dim>length(s)
    disp('Error: dimension number out of bounds. Returning...')
    return
end
n = s(dim);
nf = prod(s(1:dim-1));
nb = prod(s(dim+1:end));
t = reshape (a, [nf n nb]);
b = zeros ([nf m nb]);
for j = 1:m
    i0 = (j-1) * n / m + 1;
    i1 = j * n / m + 1;
    for i = fix(i0):min(fix(i1),n)
        b(:,j,:) = b(:,j,:) + (min(i1-1,i) - max(i0-1,i-1)) * t(:,i,:);
    end
end
s(dim) = m;
b = reshape (b, s);
%
end

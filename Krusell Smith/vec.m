function x = vec(X)
% A function to implement the vec operator, which is just a shortcut.
%
% Input:
%     X : m x n double, any matrix
% Output:
%     x : mn x 1 double, columns of X stacked

[m, n] = size(X);
x = reshape(X, [m*n 1]);
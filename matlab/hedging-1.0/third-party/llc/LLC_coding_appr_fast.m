% ========================================================================
% USAGE: [Coeff]=LLC_coding_appr(B,X,knn,lambda)
% Approximated Locality-constraint Linear Coding
%
% Inputs
%       B       -M x d codebook, M entries in a d-dim space
%       X       -N x d matrix, N data points in a d-dim space
%       knn     -number of nearest neighboring
%       lambda  -regulerization to improve condition
%
% Outputs
%       Coeff   -N x M matrix, each row is a code for corresponding X
%
% Jinjun Wang, march 19, 2010
% ========================================================================

function [Coeff] = LLC_coding_appr_fast(B, X, knn, beta, flann_options)

    if ~exist('knn', 'var') || isempty(knn)
        knn = 5;
    end

    if ~exist('beta', 'var') || isempty(beta)
        beta = 1e-4;
    end

    if ~exist('flann_options', 'var') || isempty(flann_options)
        use_flann = 0;
    else
        disp('use flann');
        use_flann = 1;
    end

    nframe=size(X,1);
    nbase=size(B,1);

    B_norm2 = sum(B.^2,2);
    X_norm2 = sum(X.^2,2);

    D = bsxfun(@plus, - 2 * X * B',B_norm2');
    D = bsxfun(@plus, D, X_norm2)';

    IDX = zeros(nframe, knn);
    for k=1:knn
        [d, mi] = min(D,[], 1);
        D(sub2ind(size(D), mi, 1:size(D,2))) = inf;
        IDX(:,k) = mi';
    end


    II = eye(knn, knn);
    Coeff = sparse(nframe, nbase)';
    for i=1:nframe
        idx = IDX(i,:);
        z = B(idx,:) - repmat(X(i,:), knn, 1);           % shift ith pt to origin
        C = z*z';                                        % local covariance
        if trace(C)==0
            C = C + 1e-10;
        end
        C = C + II*beta*trace(C);                        % regularlization (K>D)
        w = C\ones(knn,1);
        w = w/sum(w);                                    % enforce sum(w)=1
                                                         %   Coeff(i,idx) = w';
        Coeff(idx,i) = w;
    end

    Coeff = Coeff';



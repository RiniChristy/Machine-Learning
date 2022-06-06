function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for m = 1:length(X)
    min_dist = sum((X(m,:)-centroids(1,:)).^2); % first find the distance 
    % of X(m,:) with the first cluster
    % Then compare this distance with the other k clusters 
    % and find the minimum one. 
    % We can also set minimum to some random higher value, like min_dist =
    % inf or 10^6
    for k = 1:K
        dist = sum((X(m,:)-centroids(k,:)).^2); % find all the distances 
        % between X(m,:) and rest of the clusters. 
        % If the distance found is lower compared to the minimum distance set,
        % then set the new minimum distance as this distance. 
        if dist <= min_dist
            min_dist = dist;
            idx(m) = k;
        end
    end
end
% =============================================================
end


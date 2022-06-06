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


% Create a "distance" matrix of size (m x K) and initialize it to all zeros. 
% 'm' is the number of training examples, K is the number of centroids.
% Use a for-loop over the 1:K centroids.
% Inside this loop, create a column vector of the distance from each training example 
% to that centroid, and store it as a column of the distance matrix.
% Steps to be followed:
% First find the distance of X(m,:) with the first cluster.
% We can also set minimum to some random higher value, like min_dist=inf or 10^6
% Then compare this distance with the other k clusters and find the minimum one. 
% For this, find all the distances between X(m,:) and rest of the clusters. 
% If the distance found is lower compared to the minimum distance set,
% then set the new minimum distance as this distance.
% Assign the index of that training example m to that cluster with minimum distance

for m = 1:length(X)
    min_dist = sum((X(m,:)-centroids(1,:)).^2); 
    for k = 1:K
        dist = sum((X(m,:)-centroids(k,:)).^2); 
        if dist <= min_dist
            min_dist = dist;
            idx(m) = k;
        end
    end
end
% =============================================================
end




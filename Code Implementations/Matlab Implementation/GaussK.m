function [Output] = GaussK(Input,Centers)

sigma=0.2^2;
Output=exp(-(pdist2(Centers',Input,'squaredeuclidean')./sigma));
end


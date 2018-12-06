function [w,b] = InitilizeNetwork(I,alpha1,alpha2,outputNeuron,Centers)

[Y1,K1,K2]=MainKernel(alpha1,alpha2,I(1,:),Centers);
w=zeros(outputNeuron,size(Y1,2)); %Initilze Weight Matrix
b=zeros(outputNeuron,1);          % Initilze Bias Matrix

end


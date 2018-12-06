function [error,Y1,Y2,KC,KG] = Forward(I,alpha1,alpha2,w,b,Centers,d)

[Y1,KC,KG]=MainKernel(alpha1,alpha2,I,Centers);
Y2=w*Y1'+b;
error=d-Y2;

end


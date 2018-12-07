function [Output,KC,KG] = MainKernel(alpha1,alpha2,Input,Centers)

aplha11=abs(alpha1);
alpha22=abs(alpha2);
KC=CosK(Input,Centers);
KG=GaussK(Input,Centers);
Output=((aplha11*KC)+(alpha22*KG))/(aplha11+alpha22);
Output=Output';
end

 
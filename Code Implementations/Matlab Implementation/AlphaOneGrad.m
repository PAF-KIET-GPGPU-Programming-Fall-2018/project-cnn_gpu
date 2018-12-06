function [Output] = AlphaOneGrad(error,w,alpha1,alpha2,KC,KG)

Output=w*(KC-KG);
Output=Output*((abs(alpha1)*abs(alpha2))/(alpha1*(abs(alpha1)+abs(alpha2))^2));
Output=Output'*(error);
end


function [Output] = AlphaSecGrad(error,w,alpha1,alpha2,KC,KG)
Output=w*(KG-KC);
Output=Output*((abs(alpha1)*abs(alpha2))/(alpha2*(abs(alpha1)+abs(alpha2))^2));
Output=Output'*(error);

end


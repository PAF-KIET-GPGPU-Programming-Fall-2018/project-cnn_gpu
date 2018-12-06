function [Output] = CosK(Input,Centers)

gama=1e-50;
Output=(Centers'*Input');
Output=Output./(sqrt((Input*Input')*sum(Centers'.^2,2))+gama);

end


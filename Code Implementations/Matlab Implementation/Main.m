clc
clear all
close all
%%%%%%%%%%%%%%%% Data Generation Area %%%%%%%%%%%%%%%%%%
train_x=-1:0.2:1;
train_y=train_x;
test_x=-0.9:0.2:0.9;
test_y=test_x;
count=1;
for i=1:size(train_x,2)
    for j=1:size(train_x,2)
        
        TrainInput(count,1)=train_x(j);
        TrainInput(count,2)=train_y(i);
        TrainOutput(count,1)=exp(-train_x(j)^2-train_y(i));
        count=count+1;
    end
end
count=1;

for i=1:size(test_x,2)
    for j=1:size(test_x,2)
        TestInput(count,1)=test_x(j);
        TestInput(count,2)=test_y(i);
        TestOutput(count,1)=exp(-test_x(j)^2-test_y(i));
        count=count+1;
    end
end
Centers=subclust(TrainInput',0.1); % Subtractive Clustering Implemented with 121 Centers

%%
alpha1=0.5;%initilze alpha1
alpha2=0.5;%initilze alpha2
outputNeuron=1; % Number of output Neuron
[w_fused,b_fused]=InitilizeNetwork(TrainInput,alpha1,alpha2,outputNeuron,Centers);  % initialize network with weight and bias vectors


learningRate=1e-3; % Initilize Learning Rate
epoch=10000;     % number of Epoch
preAlpha1=alpha1; % strore previous Value of alpha1
preAlpha2=alpha2;% strore previous Value of alpha2
e=zeros(4,size(TrainInput,1));
et=zeros(epoch,4);
tic
for z=1:epoch
    for i=1:(size(TrainInput,1))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fussesd Kernel %%%%%%%%%%%%%%
        [error,Y1,Y2,KC,KG]=Forward(TrainInput(i,:),alpha1,alpha2,w_fused,b_fused,Centers,TrainOutput(i,:)');  % ForwardPath to netword
        e(1,i)=sum(error.^2); % Accumulate Error of Each Training Set
        w_fused=w_fused+(learningRate)*error.*Y1;   %Update Weight
        b_fused=b_fused+(learningRate)*error;       %Update Bias
        alpha1=preAlpha1+learningRate*AlphaOneGrad(error,w_fused,preAlpha1,preAlpha2,KC,KG);% Updata Alpha1 value
        alpha2=preAlpha2+learningRate*AlphaSecGrad(error,w_fused,preAlpha1,preAlpha2,KC,KG); % Updata Alpha2 Value
        alpha1=abs(alpha1)/(abs(alpha1)+abs(alpha2)); % Make Alpha1 and Alpha2 in a range under 1
        alpha2=abs(alpha2)/(abs(alpha1)+abs(alpha2));
        preAlpha1=alpha1;% hold previous value of alpha1
        preAlpha2=alpha2;% hold previouds value of alpha 2
        
       
    end
    
    et(z,1)=db(sum(e(1,:)),'power'); % finding MSE of Fused Kernel on Each Epoch in dB
   
    ap1(z)=alpha1;            % alpha1 value on Each Epoch
    ap2(z)=alpha2;            % alpha2 value on Each Epoch
end
toc
%%%%%%%%%%%%%%%%%%%%%%%% PLOTING AREA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(et(:,1),'Color','green','LineWidth',2);

grid on;
xlabel('Epoch');
ylabel('RMS(db)');
legend('Dynamic Fussed Kernel');

figure(2)
plot(ap1,'--','LineWidth',2,'Color','red');
hold on;
plot(ap2,'--','LineWidth',2,'Color','blue');
grid on;
hold off;
xlabel('Epoch');
legend('a1 Cosine','a2 Gauss');

%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%%%%%%%%%%%

acumulator=zeros(4,1);
for k=1:size(TestInput,1)
    
    [error,Y1,Y2,KC,KG]=Forward(TestInput(k,:),alpha1,alpha2,w_fused,b_fused,Centers,TestOutput(k,:)'); %Send Input to Network To Get Output
    acumulator(1,k)=Y2;
    
   
    
end

figure(3)

plot(acumulator(1,:),'LineWidth',2,'Color','green');
grid on;
legend('Proposed dynamic Fusion of Gauss & Cosine Kernels');
hold off
alpha1
alpha2

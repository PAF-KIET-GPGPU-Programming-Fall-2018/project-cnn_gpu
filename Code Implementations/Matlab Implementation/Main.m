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

w_cosine=w_fused;   % Initilize cosine Weight
b_cosine=b_fused;   % Initilize cosine Bias

w_gauss=w_fused;    % Initilize Guass Weight
b_gauss=b_fused;    % Initilize Gause Bias

w_manual=w_fused;   % Initilize Manual Weight
b_manual=b_fused;   % Initilize Manual Bias

learningRate=1e-3; % Initilize Learning Rate
epoch=10000;     % number of Epoch
preAlpha1=alpha1; % strore previous Value of alpha1
preAlpha2=alpha2;% strore previous Value of alpha2
e=zeros(4,size(TrainInput,1));
et=zeros(epoch,4);

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
        
        %%%%%%%%%%%%%%%%%%%%%% Cosine Kernel %%%%%%%%%%%%%%%%%%%%%%%
        
        %For Cosine Kernel Alpha1=1 and Alpha2=0, put in below equation
        [error,Y1,Y2,KC,KG]=Forward(TrainInput(i,:),1,0,w_cosine,b_cosine,Centers,TrainOutput(i,:)');  % Forward Path
        e(2,i)=sum(error.^2); % Accumulate Error of Each Training Set
        w_cosine=w_cosine+(learningRate)*error.*Y1;   %Update Weight
        b_cosine=b_cosine+(learningRate)*error;       %Update Bias
        
        %%%%%%%%%%%%%%%%%%%%%% Gauss Kernel %%%%%%%%%%%%%%%%%%%%%%%%
        %For Gauss Kernel Alpha1=0 and Alpha2=1, put in below equation
        [error,Y1,Y2,KC,KG]=Forward(TrainInput(i,:),0,1,w_gauss,b_gauss,Centers,TrainOutput(i,:)');  % ForwardPath to netword
        e(3,i)=sum(error.^2); % Accumulate Error of Each Training Set
        w_gauss=w_gauss+(learningRate)*error.*Y1;   %Update Weight
        b_gauss=b_gauss+(learningRate)*error;       %Update Bias
        
        %%%%%%%%%%%%%%%%%%%% Manualy Fussed %%%%%%%%%%%%%%%%%%%
        % for Manual we put alpha1=0.5 and alpha2=0.5
        [error,Y1,Y2,KC,KG]=Forward(TrainInput(i,:),0.5,0.5,w_manual,b_manual,Centers,TrainOutput(i,:)');  % ForwardPath to netword
        e(4,i)=sum(error.^2); % Accumulate Error of Each Training Set
        w_manual=w_manual+(learningRate)*error.*Y1;   %Update Weight
        b_manual=b_manual+(learningRate)*error;       %Update Bias
    end
    
    et(z,1)=db(sum(e(1,:)),'power'); % finding MSE of Fused Kernel on Each Epoch in dB
    et(z,2)=db(sum(e(2,:)),'power'); % finding MSE of Cosine Kernel on Each Epoch in dB
    et(z,3)=db(sum(e(3,:)),'power'); % finding MSE of Gauss Kernel on Each Epoch in dB
    et(z,4)=db(sum(e(4,:)),'power'); % finding MSE of Manual Kernel on Each Epoch in dB
    %et(z)=mse(e); % finding MSE on Each Epoch in dB
    ap1(z)=alpha1;            % alpha1 value on Each Epoch
    ap2(z)=alpha2;            % alpha2 value on Each Epoch
end
%%%%%%%%%%%%%%%%%%%%%%%% PLOTING AREA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(et(:,1),'Color','green','LineWidth',2);
hold on;
plot(et(:,2),'--','Color','red','LineWidth',2);
plot(et(:,3),'--','Color','blue','LineWidth',2);
plot(et(:,4),'-','Color','black','LineWidth',2);
grid on;
xlabel('Epoch');
ylabel('RMS(db)');
legend('Dynamic Fussed Kernel','Cosine Kernel', 'Gauss Kernel','Manual');

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
    
    [error,Y1,Y2,KC,KG]=Forward(TestInput(k,:),1,0,w_cosine,b_cosine,Centers,TestOutput(k,:)'); %Send Input to Network To Get Output
    acumulator(2,k)=Y2;
    
    [error,Y1,Y2,KC,KG]=Forward(TestInput(k,:),0,1,w_gauss,b_gauss,Centers,TestOutput(k,:)'); %Send Input to Network To Get Output
    acumulator(3,k)=Y2;
    
    [error,Y1,Y2,KC,KG]=Forward(TestInput(k,:),0.5,.5,w_manual,b_manual,Centers,TestOutput(k,:)'); %Send Input to Network To Get Output
    acumulator(4,k)=Y2;
    
end

figure(3)
plot(acumulator(3,:),'--','LineWidth',2,'Color','blue');
hold on;
plot(acumulator(2,:),'--','LineWidth',2,'Color','red');
plot(acumulator(4,:),'-.','LineWidth',2,'Color','black');
plot(TestOutput','LineWidth',2,'Color','Yellow');
plot(acumulator(1,:),'LineWidth',2,'Color','green');
grid on;
legend('Gauss Kernel based RBF','Cosine Kernel based RRBF','Manual fusion of Gauss & Cosine Kernels', 'Actual Output','Proposed dynamic Fusion of Gauss & Cosine Kernels');
hold off
alpha1
alpha2

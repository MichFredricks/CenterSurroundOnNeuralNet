results = zeros(9,100);

%%
load('MNIST.mat');

fprintf('\n');
fprintf('Using MNIST hand written digit data set with no filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork2d(X,y,false,[20,20],20,false));

%Apply Center Surround Receptive Field
fprintf('Using MNIST hand written digit data set with Center-On Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork2d(X,y,true,[20,20],20,true));

fprintf('Using MNIST hand written digit data set with Center-Off Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork2d(X,y,true,[20,20],20,false));
%%

%repeated 100 times: 
for i = 1:100
    results(1,i) = neuralnetwork2d(X,y,false,[20,20],20,false);
    results(2,i) = neuralnetwork2d(X,y,true,[20,20],20,true);
    results(3,i) = neuralnetwork2d(X,y,true,[20,20],20,false);
end
%%
load('ovariancancerdata.mat');
% 
fprintf('Using Ovarian Cancer predicter data set with no filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,false,20,false));
fprintf('Using Ovarian Cancer predicter data set with Center-On Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,true,20,true));
fprintf('Using Ovarian Cancer predicter data set with Center-Off Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,true,20,false));
%%
%repeated 100 times: 
for i = 1:100
    results(4,i) = neuralnetwork1d(X,y,false,20,false);
    results(5,i) = neuralnetwork1d(X,y,true,20,true);
    results(6,i) = neuralnetwork1d(X,y,true,20,false);
end
%%
load('arcene.mat');

fprintf('Using arcene with no filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,false,20,false));
fprintf('Using arcene with Center-On Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,true,20,true));
fprintf('Using arcene with Center-Off Surround filter:\n');
fprintf('Training Set Accuracy: %f\n', neuralnetwork1d(X,y,true,20,false));
%%
for i = 1:100
    results(7,i) = neuralnetwork1d(X,y,false,20,false);
    results(8,i) = neuralnetwork1d(X,y,true,20,true);
    results(9,i) = neuralnetwork1d(X,y,true,20,false);
end
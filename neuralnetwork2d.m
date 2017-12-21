function TrainingAccuracy = neuralnetwork(X,y,receptive_field,dimensions,hidden_layer_size,center_on)
%%
input_layer_size  = size(X,2);  % 20x20 Input Images of Digits
num_labels = max(y);          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%Apply Center Surround Receptive Field
if receptive_field
   if center_on
       load('center_on_receptive_field_2d.mat');
   else
       load('center_off_receptive_field_2d.mat');
   end
   
    for i = 1:size(X,1)
        rect= reshape(X(i,:),dimensions);
        rect = filter2(weights,rect);
%         temp = zeros(dimensions+6);
%         temp(4:dimensions(1)+3,4:dimensions(2)+3) = rect;
%         for j = 1:dimensions(1)
%             for k = 1:dimensions(2)
%                 out = sum(sum(temp(j:j+6,k:k+6).*weights));
%                 rect(j,k) = out;
%             end
%         end
        rect(rect<0) = 0;
        X(i,:) = rect(:);
    end
    
end
%%
shuffle = randperm(size(X,1))';
X = X(shuffle,:);
y = y(shuffle,:);
Xtest = X(end-(floor(size(X,1)/10))+1:end,:);
ytest = y(end-floor(length(y)/10)+1:end);
X = X(1:end-(floor(size(X,1)/10)),:);
y = y(1:end-(floor(length(y)/10)));

m = size(X, 1);


%Evaluate sigmoidGradient
g = sigmoidGradient([-1 -0.5 0 0.5 1]);

% randomly initialize weights 

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Minimize cost
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);

TrainingAccuracy = mean(double(pred == y)) * 100;
end


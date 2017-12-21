function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1), X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];
z3 = Theta2*a2;
h = sigmoid(z3);

b = zeros(num_labels,m);
for i = 1:m
    b(y(i),i)= 1;
end
del3 = h - b;
z2 = [ones(1,m); z2];
del2 = Theta2'*del3 .* sigmoidGradient(z2);
Theta2_grad = del3*a2';
Theta1_grad = del2*X;
Theta1_grad = Theta1_grad(2:end,:);
Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;
for i = 1:m
    b(:,i) = (-b(:,i)).*log(h(:,i))-(1-b(:,i)).*log(1-h(:,i));
end
J = sum(b(:))/m;
Theta1Reg = Theta1(:,1);
Theta2Reg = Theta2(:,1);
R = lambda*(sum(Theta1(:).^2)-sum(Theta1Reg(:).^2)+sum(Theta2(:).^2)-sum(Theta2Reg(:).^2))/(2*m);
J = sum(b(:))/m + R;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

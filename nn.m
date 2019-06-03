max_val = 255;
rand_mult_w = 0.01;
rand_mult_b = 0.00001;
X = transpose(double(trainX(1,:))/max_val);
Y = zeros(10,1);
Y(trainY(1))=1;

ain = size(X);
aon = size(Y);
ahn = 700;

w_2 = rand(ahn, ain(1))*rand_mult_w;
w_3 = rand(aon(1), ahn)*rand_mult_w;
b_2 = rand(ahn,1)*rand_mult_b;
b_3 = rand(aon(1),1)*rand_mult_b;

z_2 = zeros(ahn, 1);
z_3 = zeros(aon(1), 1);

a_1 = zeros(ain(1),1);
a_2 = zeros(ahn,1);
a_3 = zeros(aon(1),1);

% feed forward
a_1 = X(:,1);
z_2 = w_2*a_1+b_2;
a_2 = logsig(z_2);
z_3 = w_3*a_2+b_3;
a_3 = logsig(z_3)

delta_3 = 

% backprop


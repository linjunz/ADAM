%% Adaptive Linear Discriminant Analysis with missing data
% It is based on the constrained l_1 minimization to classify a
% p-dimensional vecotr to one of two multivariate Gaussian distributions
% with the same covariance matrix based on the observed data with missing positions.

%% Outputs:
% error: estimated parameters for the Gaussian mixtures
% IDX: vector of class membership
% Inputs: 
% xt: n1 by p training data from class 1
% yt: n2 by p training data from class 2
% S1: n1 by p 0-1 matrix denoting the missing positions in xt
% S2: n2 by p 0-1 matrix denoting the missing positions in yt
% ztest: n_z by p testing data drawn from one of two classes
% label_z: true labels of the test data, used for evaluating the
%   classificatoin performance.
% lambda0: a scalar, the tuning parameter for estimating sparse beta. Default
%    is 8. Different choice may lead to better finite sample peformance.
%
function [error, IDX] = ADAM(xt,yt,S1,S2,ztest,label_z,lambda0=8)
[n1,p]=size(xt);
[n2,p]=size(yt);
%% generalized sample means
n_mat1=S1'*S1;
n_mat2=S2'*S2;
mu_mat1=xt'*S1*diag(diag(n_mat1).^(-1));
hatmux=diag(mu_mat1);
mu_mat2=yt'*S2*diag(diag(n_mat2).^(-1));
hatmuy=diag(mu_mat2);
mu = [hatmux, hatmuy];
hatdelta=hatmux-hatmuy;
%% generalized sample covariance matrix
xt_new = xt-ones(n1,1)*hatmux';
xt_new = S1.* xt_new;
yt_new = yt-ones(n2,1)*hatmuy';
yt_new = S2.* yt_new;
hatSigma =((n_mat1+n_mat2).^(-1)).*(xt_new'*xt_new+yt_new'*yt_new);
%% Step 1 of the ADAM procedure
d=diag(hatSigma);
n=floor((n1+n2)/2);
a=lambda0*sqrt(log(p)/n)*(sqrt(d));%t1 --2
B=sqrt(log(p)/n)*sqrt(d)*hatdelta'*2;%t2 --1.5
f=ones(2*p,1);
CoeffM=[hatSigma-B -(hatSigma-B);-(hatSigma+B) hatSigma+B];
Coeffb=[a+hatdelta; a-hatdelta];
options=optimoptions('linprog','Algorithm','interior-point','Display','none');
uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*p,1),[],[],options);
beta0=uv(1:p)-uv((p+1):(2*p));
%% Step 2 of the ADAM procedure
lambda=[];
for k=1:p
    lambda=[lambda, 2*sqrt(log(p)/n)*sqrt(hatSigma(k,k)*(abs(beta0'*hatdelta)+2))];
end
lambda=lambda*1;%t3 --1.8
CoeffMnew=[hatSigma -(hatSigma);-(hatSigma) hatSigma];
Coeffbnew=[lambda'+hatdelta;lambda'-hatdelta];
options=optimoptions('linprog','Algorithm','interior-point');
uv_new=linprog(f,CoeffMnew,Coeffbnew,[],[],zeros(2*p,1),[],[],options);
beta=uv_new(1:p)-uv_new((p+1):(2*p));
%% Evaluation on the testing data
IDX_v=(ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta);
IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta <= 1e-06 ) + 1; %classification
error=sum(abs(IDX-label_z))/size(ztest,1);
end
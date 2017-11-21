function error = ALPD(xt,yt,ztest,label_z)
%%
[n,p]=size(xt);
hatmux=mean(xt);  %estimation of mean
hatmuy=mean(yt);
mu = [hatmux', hatmuy'];
hatdelta=hatmux'-hatmuy';
hatSigma=(cov(xt)+cov(yt))/2; %estimation of covariance matrix
lambda0=1;
%%

d=diag(hatSigma);
%a=sqrt(log(p)/n)*(sqrt(d)+abs(hatdelta).^2);
a=1*sqrt(log(p)/n)*(sqrt(d));
%%
B=lambda0*sqrt(log(p)/n)*sqrt(d)*hatdelta';
f=ones(2*p,1);
CoeffM=[hatSigma-B -(hatSigma-B);-(hatSigma+B) hatSigma+B];
Coeffb=[a+hatdelta; a-hatdelta];
options=optimoptions('linprog','Algorithm','interior-point','Display','none');
uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*p,1),[],[],options);
beta0=uv(1:p)-uv((p+1):(2*p));
%%
lambda=[];
for k=1:p
    %lambda=[lambda sqrt(log(p)/n)*sqrt(hatSigma(k,k)*(abs(beta0'*hatdelta)+1)+hatdelta(k)^2)];
    lambda=[lambda, 1*sqrt(log(p)/n)*sqrt(lambda0*hatSigma(k,k)*(abs(beta0'*hatdelta)+1))];
end
CoeffMnew=[hatSigma -(hatSigma);-(hatSigma) hatSigma];
Coeffbnew=[lambda'+hatdelta;lambda'-hatdelta];
uv_new=linprog(f,CoeffMnew,Coeffbnew,[],[],zeros(2*p,1),[],[],options);
beta=uv_new(1:p)-uv_new((p+1):(2*p));

IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta <=1e-06 ) + 1; %classification
error=sum(abs(IDX-label_z))/size(ztest,1);
end
function error = ALARM(xt,yt,S1,S2,ztest,label_z)
%%
[n1,p]=size(xt);
[n2,p]=size(yt);
n_mat1=S1'*S1;
n_mat2=S2'*S2;
mu_mat1=xt'*S1*diag(diag(n_mat1).^(-1));
hatmux=diag(mu_mat1);
mu_mat2=yt'*S2*diag(diag(n_mat2).^(-1));
hatmuy=diag(mu_mat2);
mu = [hatmux, hatmuy];
hatdelta=hatmux-hatmuy;
%%
xt_new = xt-ones(n1,1)*hatmux';
xt_new = S1.* xt_new;
yt_new = yt-ones(n2,1)*hatmuy';
yt_new = S2.* yt_new;
hatSigma =((n_mat1+n_mat2).^(-1)).*(xt_new'*xt_new+yt_new'*yt_new);
d=diag(hatSigma);
n=floor((n1+n2)/2);
%a=sqrt(log(p)/n)*(sqrt(d)+abs(hatdelta).^2);
a=8*sqrt(log(p)/n)*(sqrt(d));%t1 --2
%%
B=sqrt(log(p)/n)*sqrt(d)*hatdelta'*2;%t2 --1.5
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
    lambda=[lambda, 2*sqrt(log(p)/n)*sqrt(hatSigma(k,k)*(abs(beta0'*hatdelta)+2))];
end
lambda=lambda*1;%t3 --1.8
CoeffMnew=[hatSigma -(hatSigma);-(hatSigma) hatSigma];
Coeffbnew=[lambda'+hatdelta;lambda'-hatdelta];
%options = optimset('Display','none');
%%
options=optimoptions('linprog','Algorithm','interior-point');
uv_new=linprog(f,CoeffMnew,Coeffbnew,[],[],zeros(2*p,1),[],[],options);
%uv_new=linprog(f,CoeffMnew,Coeffbnew,[],[],zeros(2*p,1));
beta=uv_new(1:p)-uv_new((p+1):(2*p));
%%
w=n2/(n1+n2);
%w=15/149;
IDX_v=(ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta -0* log (w/(1-w));
IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta +0* log (w/(1-w)) <= 1e-06 ) + 1; %classification
error=sum(abs(IDX-label_z))/size(ztest,1)
end
function [G, A,S, Loss] = LSTD_aaai(X, index, gt, param)
%   min    alpha*|A|_{t*} + beta*|S|_1 + delta*|C|_{t1} + |E|_{2,1} 
%   s.t.  Xi = Xi*Zi + E^i , O'ZiO' = Bi+Si, A = B, B=C
%
num_view = length(X);
N = numel(gt);
dk=cell(1, num_view); 
for i=1:num_view
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);  %normalized
    dk{i} = size(X{i},1);
    num{i} = size(X{i}, 2);
end

O = cell(1, num_view);
for i=1:num_view
    pos = find(index(:,i)==1);
    T= zeros(N, length(pos));
    for j=1:length(pos)
        T(pos(j),j) = 1;     
    end
    O{i} = T;
end
lambda  = param.lambda;
theta   =  param.theta;
mu = param.mu;

for iv = 1:num_view
    A{iv} = zeros(size(X{iv},2));
    A{iv} = O{iv}*A{iv}*O{iv}';
    E{iv} = zeros(dk{iv}, num{iv});
    Z{iv} = zeros(num{iv});
    F2{iv} = zeros(N);
end
F1 = E;
S = F2;
B = A;
C = A;
F3 = F2;
F4 = F3;
MAX_iter = 30;

rho = 1e-3;
dt = 1.3;

for iter = 1:MAX_iter
     if mod(iter, 10)==0
      fprintf('%d..',iter);
     end
     
     % Zi
     for i=1:num_view
        OiO=  O{i}'*O{i};
        tempM = X{i}-E{i}+F1{i}/rho;
        tempN = B{i} + S{i} -  F2{i}/rho;
        tempA = OiO\(X{i}'*X{i});
        tempC = OiO\(X{i}'*tempM +O{i}'*tempN*O{i});
        Z{i} = sylvester(tempA, OiO, tempC);
     end
     clear OiO tempM tempN tempA tempC
    
      % E step
     for i=1:num_view
        E{i} = prox_l21(X{i}-X{i}*Z{i} + F1{i}/rho, 1/rho);
     end
  
    % A step
    B_tensor = cat(3, B{:,:});
    F3_tensor = cat(3, F3{:,:});
    Bv = B_tensor(:);
    F3v = F3_tensor(:);
    [Av, ~] = wshrinkObj_nc(param.fun, param.gamma, Bv + 1/rho*F3v, lambda/rho, [N, N, num_view], 0, 3);
    %[Av, ~] = wshrinkObj(Bv + 1/rho*F3v, alpha/rho, [N, N, num_view], 0, 1);
    A_tensor = reshape(Av, [N, N, num_view]);  
    for i=1:num_view
        A{i} = A_tensor(:,:,i);
    end
    
    % S step
    for i=1:num_view
        S{i} = prox_l1(O{i}*Z{i}*O{i}'-B{i}+F2{i}/rho, theta/rho);
    end
    
    % C step
    B_tensor = cat(3, B{:,:});
    F4_tensor = cat(3, F4{:,:});
    C_tensor  = cat(3, C{:,:});
    tempBF4 = B_tensor +  F4_tensor/rho;
    for i=1:N
        C_tensor(i,:,:)  = prox_l21(reshape(tempBF4(i,:,:), [num_view, N])', mu/rho);%C_tensor(i,:,:) 
    end
    for i=1:num_view
        C{i} = C_tensor(:,:,i);
    end
    
    
    % B
    for i=1:num_view
        B{i} = (O{i}*Z{i}*O{i}'-S{i} + F2{i}/rho + A{i} + F3{i}/rho + C{i} - F4{i}/rho)/3;
    end

    
    %
    RR1 = [];RR2=[]; RR3=[];RR4=[];
    for i=1:num_view
        res1 = X{i} - X{i}*Z{i} - E{i};
        res2 = O{i}*Z{i}*O{i}'-B{i} -S{i};
        res3 = A{i} - B{i};
        res4 = B{i} - C{i};
        F1{i} = F1{i} + rho*res1;
        F2{i} = F2{i} + rho*res2;
        F3{i} = F3{i} + rho*res3;
         F4{i} = F4{i} + rho*res4;
        RR1 =[RR1, norm(res1,'inf')];
        RR2 =[RR2,  norm(res2,'inf')];
        RR3 =[RR3,  norm(res3,'inf')];
        RR4 =[RR4,  norm(res4,'inf')];
    end
 
    rho = min(1e6, dt*rho);
    
%     loss1(iter) = norm(RR, inf);
%     loss2(iter) = norm(H-H*Q{num_view+1}-E2,inf);
%     loss3(iter) = norm(RR2, inf);
%     loss4(iter) = norm(RR3, inf);
%     loss5(iter) = norm(RR4, inf);
    thrsh = 1e-5;
    if(norm(RR1, inf)<thrsh && norm(RR2, inf)<thrsh && norm(RR3, inf)<thrsh )
        break;
    end
  
    loss1(iter) = norm(RR1, inf);     loss2(iter) = norm(RR2, inf);
    RR1 = [];RR2=[]; RR3=[];RR4=[];
end

KK=0;
 for i=1:num_view
    KK = KK + (abs(A{i})+(abs(A{i}))');
    %KK = KK + A{i}+(A{i})';
 end
G = KK/2/num_view; 
Loss= [loss1; loss2];

end

clear; close all;

addpath ./ClusteringMeasure
addpath ./nonconvex_funs
path = './data/';


load  ./data/ORL
name = 'ORL';
percentDel  = 0.1; 
Datafold    =  [path,'Index_',name,'_percentDel_',num2str(percentDel),'.mat'];
load(Datafold)

param.lambda  = 10;   
param.theta  = 1;    
param.mu  = 1;    
param.fun    = 'laplace'; param.gamma = 10;
cls_num = numel(unique(Y));

perf = []; gt = double(Y);
for kk = 1:length(Index) 
    Xc = X;
    ind = Index{kk};
    for i=1:length(Xc)
        Xci = Xc{i};
        indi = ind(:,i);
        pos = find(indi==0);
        Xci(:,pos)=[]; 
        Xc{i} = Xci;
    end   
    
    G = ETLSRR(Xc, ind, Y, param);
    
   for rp = 1:10 % 10 runs of kmeans
        [Clus] = SpectralClustering(G, cls_num);
        [ACC,NMI,PUR] = ClusteringMeasure(gt,Clus); %ACC NMI Purity
        [Fscore,Precision,R] = compute_f(gt,Clus);
        [AR,~,~,~]=RandIndex(gt,Clus);
        result = [ACC NMI AR Fscore PUR Precision R];
        perf  = [perf; result*100];
        fprintf("ITER: %d, ACC,NMI, ARI: %.4f, %.4f, %.4f, \n", kk, result(1),result(2),result(3));
   end
end





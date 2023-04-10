function [confusion_matrix,trace_max]=confusion_compute(label_predict,num_each_class)
%% compute the confusion matrix
% Input:
%       label_predict, 1*num_sample vector, the predicted label, ranges from 1 to num_cluster
%       num_each_class, 1*num_class vector, the size of each class in the ground truth 
% Output:
%       confusion_matrix, num_cluster*num_cluster matrix, each row corresponds to each predicted cluster
%       trace_max, the maximal value of trace

 num_cluster=length(num_each_class);
 
 % determine the order of predicted labels, such as (1 2 3), or (2 3 1)
% num_accumulation=0;
% order_label=zeros(1,num_cluster);  
% for k=1:num_cluster
%     size_1=sum(label_predict(num_accumulation+1:num_accumulation+num_each_class(k))==1);
%     size_2=sum(label_predict(num_accumulation+1:num_accumulation+num_each_class(k))==2);
%     size_3=sum(label_predict(num_accumulation+1:num_accumulation+num_each_class(k))==3);
%     [~,order_label(k)]=max([size_1 size_2 size_3]);
%     num_accumulation=num_accumulation+num_each_class(k);
% end
 
confusion_matrix=zeros(num_cluster,num_cluster);

num_accumulation=0;
for i=1:num_cluster
    for j=1:num_cluster
        confusion_matrix(i,j)=sum(label_predict(num_accumulation+1:num_accumulation+num_each_class(i))==j);  %order_label(j)
    end
    num_accumulation=num_accumulation+num_each_class(i);
end

%% search for the largest trace of the confusion matrix
location_index=perms([1:num_cluster]);
num_permutation=size(location_index,1);
trace_value=zeros(1,num_permutation);
for i=1:num_permutation
    for j=1:num_cluster
        trace_value(i)=trace_value(i)+confusion_matrix(location_index(i,j),j);
    end
end

[trace_max,location_max]=max(trace_value);
       
confusion_new=confusion_matrix(location_index(location_max,:),:);
confusion_matrix=confusion_new; 
        
            
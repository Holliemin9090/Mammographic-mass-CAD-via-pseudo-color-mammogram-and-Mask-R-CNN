function [num_tp,num_fp,Dice] = TPR_FPR(all_detect,all_anno)
%TPR_FPR Summary of this function goes here
%   Detailed explanation goes here
%   Caculate the number of the TPs(true positive) and FPs(false positive)
%   INPUT    all_detect    a struct containing all the connected objects
%                          detected
%            all_anno      a struct containing all the cnnected objects in
%                          the annotation mask
%   OUTPUT   num_tp        number of true positives
%            num_fp        number of false positives
%            Dice          dice similarity index between the segmentation
%                          and the annotation

anno_ind = zeros(1,length(all_anno));

num_tp = 0;
num_fp = 0;

Dice =zeros(max(1,length(all_detect)),length(all_anno));

for i = 1:length(all_detect)
    msk = labelmatrix(all_detect{i});
    msk(msk>1) = 1;
    msk = double(msk);
    tp_flag = 0;
    for j = 1:length(all_anno)
        anno = labelmatrix(all_anno{j});
        anno(anno>1) = 1;
        anno = double(anno);
        % calculate dice similarity
               
        [D] = calculate_dice(msk,anno);
        Dice(i,j) = D;
        
        if D>=0.2
            anno_ind(j) = 1;
            tp_flag = 1;% this detection is a true positive
        end
    end
    
    if tp_flag==0
        num_fp = num_fp+1;% count the false positive
    end
    
end

% count the true positive
num_tp = num_tp+length(find(anno_ind==1));
Dice = max(Dice,[],1);
end


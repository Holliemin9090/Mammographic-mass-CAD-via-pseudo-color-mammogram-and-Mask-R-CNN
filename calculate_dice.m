function [DICE] = calculate_dice(BW1,BW2)
%CALCULATE_DICE Summary of this function goes here
%   Detailed explanation goes here

non_zero1 = find(BW1(:) >0);
non_zero2 = find(BW2(:) >0);

Area1 = length(non_zero1);
Area2 = length(non_zero2);

temp = ismember(non_zero1,non_zero2);
intersection = length(find(temp>0));

DICE = 2*intersection/(Area1+Area2);
end


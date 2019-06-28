function [new_im] = Normalization_mask(image,mask,mode)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
image = double(image);
inten = image(mask==1);
mini = min(inten);
maxi = max(inten);

image = (image-mini)./abs(maxi-mini);
image(mask<1) = 0;
if mode == 8
    new_im = uint8(image*255);
    
elseif mode==16
    new_im = uint16(image*65535);
    
elseif strcmp(mode,'double')
    new_im = double(image);
end
end


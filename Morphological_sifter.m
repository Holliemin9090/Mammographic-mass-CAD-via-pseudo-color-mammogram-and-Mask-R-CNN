function [enhanced_image]= Morphological_sifter(M1,M2,orientation,image,L_or_R, padding_option, breast_mask)%,rowmin,colmin
% Summary of this function goes here
%   Detailed explanation goes here
% This is a function that does multi-scale morphological sifting used
% linear structuring elements (LSE) with given length
%INPUT  M1,M2: Length of the LSEs  M1>M2
%       orientation: Orientations of the LSEs in degrees, it is a 1*N
%       vector containing the angles of each LSE
%       image: The input image to be processed
%       L_or_R: Indicator of left or right breast
%       padding_option: Image boundary padding options. 
%                       If set to 0, pad the boundary with highest
%                       intensity value.
%                       If set to 1, pad it with replications of the pixels
%                       on the boundary
%       breast_mask: The binary breast mask
%OUTPUT enhanced_image: The output image from MMS
%       


newimage = image;
[m,n]=size(newimage);

%% Border effect control: border padding
% Option 1: pad with highest pixel value
temp = uint16(65535*ones(m+4*M1,n+4*M1));
temp(2*M1+1:2*M1+m,2*M1+1:2*M1+n) = newimage; % Add white margins to each side of the image to prevent edge effect of morphological process

% Option 2: replicate the pixels on the border
if padding_option == 1
    if L_or_R == 1% left breast
        edge = newimage(:,1:min(n,2*M1)) ;
        temp(2*M1+1:2*M1+m,2*M1-size(edge,2)+1:2*M1)= fliplr(edge);
    else % right breast
        edge = newimage(:,max(1,n-2*M1+1):n) ;
        temp(2*M1+1:2*M1+m,n+2*M1+1:n+2*M1+size(edge,2))= fliplr(edge);
    end
end
%% Apply multi-scale morphological sifting

enhanced_image = zeros(size(temp));
for k = 1:length(orientation)
    B1=strel('line',M1,orientation(k));
    B2=strel('line',M2,orientation(k));
    bg1=imopen(temp,B1);
    r1=imsubtract(temp,bg1);
    r2=imopen(r1,B2);
    enhanced_image = enhanced_image + double(r2);
    
end

enhanced_image = enhanced_image(2*M1+1:2*M1+m,2*M1+1:2*M1+n); % Reset the image into the original size
[enhanced_image] = Normalization_mask(enhanced_image,breast_mask,8);

end







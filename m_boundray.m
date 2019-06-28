function [ n ] = m_boundray( marklabel,color, varargin )
%m_boundray Summary of this function goes here
%   Detailed explanation goes here
% Delineate the boundary of the mask
% INPUT     marklabel   the mask
%           color       the color of the line
%           varargin    there can be one extra input setting the line
%           width, the default line width is 1.5
% OUTPUT    n           number of connected objects in the mask

if nargin>2
    line_width = varargin{1};
else
    line_width = 1.5;% 
end

if ~isstruct(marklabel)
    [gn,n]=bwlabel(marklabel, 8);% the number of lesions is n
    hold on
    for i=1:n
        testim=zeros(size(gn));
        testim(gn==i)=1;
        
        boundary= bwboundaries(testim);%
        hold on
        for k = 1:length(boundary)
            
            plot(boundary{k}(:,2), boundary{k}(:,1),'Color',color, 'LineWidth', line_width);
        end
        
    end
else
    n = marklabel.NumObjects;
    for j = 1:marklabel.NumObjects        
        testim = zeros(marklabel.ImageSize);
        testim(marklabel.PixelIdxList{j})=1;
        boundary= bwboundaries(testim);%
        hold on
        for k = 1:length(boundary)
            
            plot(boundary{k}(:,2), boundary{k}(:,1),color, 'LineWidth', line_width);
        end
        
    end
    
end
hold off
end


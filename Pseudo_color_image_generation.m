% Try to feed in gradient and morphological filtered images in to different
% channal
clc,clear,close all;
tic

image_path = 'scans\preprocessed_image\';
image_save_path = 'scans\pseudo_color_image\';

if ~exist(image_save_path,'dir')
    mkdir(image_save_path)
end
item_names = Read_files_in_folder( image_path, 'files' );

mass_size_range_mm = [15 3689];% square mm
resolution = 0.07;% spatial resolution of the INbreast mammograms, 0.07mm
resize_ratio = 1/4;
mass_diameter_range_pixel = [floor((mass_size_range_mm(1)/pi)^0.5*2/(resolution/resize_ratio)),...
    ceil((mass_size_range_mm(2)/pi)^0.5*2/(resolution/resize_ratio))];% diameter range in pixels

for i = 1:length(item_names)
    close all;
    disp(item_names{i});
    image = imread(strcat(image_path,item_names{i}));
    
    %% Image subsampling using 2 level db2 wavelet
    image = image(:,:,1);
    breast_mask = (image>0);
    [cA,~,~,~] = dwt2(image,'db2');
    [image,~,~,~] = dwt2(cA,'db2');
    
    [cA,~,~,~] = dwt2(breast_mask,'db2');
    [breast_mask,~,~,~] = dwt2(cA,'db2');
    breast_mask = (breast_mask>=1);
    
    
    % Normalize the grayscale image
    [new_im] = Normalization_mask(image,breast_mask,8);
    %     figure,imshow(new_im);
    
    %% Apply multi-scale morphological sifting and append the images from 2 scales to the grayscale mammogram
    L_OR_R = isempty(strfind(item_names{i},'_R_'));% check if it is a left or right breast
    CC_OR_ML = isempty(strfind(item_names{i},'_CC_'));
    degree_bank = 0:10:170;% The orientations of the linear structuring elements (LSEs)
    Num_scale = 2; % Using 2 scales
    % Generate the length for LSEs on different scales
    [ len_bank ] = Morphological_filter_bank( Num_scale, mass_diameter_range_pixel, 'exponential' );
    enhanced_image = {};
    for j = 1:Num_scale
        % Boundary padding
        padding_mode = 1;%
        if j==1||CC_OR_ML==1
            %           if it is a small scale or it is a MLO view
            padding_mode = 0;% highest value padding
        end
        [enhanced_image{j}] = Morphological_sifter(len_bank(j+1),len_bank(j),degree_bank,new_im,L_OR_R, padding_mode, breast_mask);
        %
    end
    Pseudo_color_im = cat(3,new_im,enhanced_image{1},enhanced_image{2});
        figure,imshow(Pseudo_color_im);
        imwrite(Pseudo_color_im,strcat(image_save_path,item_names{i}));
end
elapsedTime = toc;

%% Process the annotation masks, so that they are the same size as the mammograms
anno_path = 'scans\preprocessed_mask\';
anno_save_path = 'scans\preprocessed_mask1\';
if ~exist(anno_save_path,'dir')
    mkdir(anno_save_path)
end

item_names = Read_files_in_folder( anno_path, 'files' );
for i = 1:length(item_names)
    anno = imread(strcat(anno_path,item_names{i}));
    anno(anno==255) = 1;
    anno= double(anno);
    [cA,~,~,~] = dwt2(anno,'db2');
    [anno,~,~,~] = dwt2(cA,'db2');
    anno = abs(anno);
    anno(anno>=1) = 255;
    anno(anno<1) = 0;
    anno = uint8(anno);
        imwrite(anno,strcat(anno_save_path,item_names{i}));
end
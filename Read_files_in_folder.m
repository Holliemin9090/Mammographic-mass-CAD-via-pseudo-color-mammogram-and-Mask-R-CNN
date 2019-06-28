function [ item_name ] = Read_files_in_folder( path, mode )
% Read_files_in_folder Summary of this function goes here
%   Detailed explanation goes here
% This is a fuction that reads the files/folders under a folder path
%   INPUT   path   the folder path
%           mode   whether the user want read files or folders
%   OUTPUT  item_name    return the names of the items under the path

pt=dir(path);

item_name = {}; 

M=length(pt);

k = 0;
format short


for i = 1 : M
    if strcmp(pt (i).name, '.') || strcmp(pt (i).name, '..')||(pt(i).isdir==0 && strcmp(mode,'folder'))
        continue;
    else
            k = k + 1;
            item_name{k} = pt (i).name;
    end
    
end


    

end


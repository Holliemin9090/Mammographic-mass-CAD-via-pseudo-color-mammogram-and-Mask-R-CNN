function [ len_bank ] = Morphological_filter_bank( Num_scale, D, type )
%Morphological_filter_bank Summary of this function goes here
%   This function generates the length of the linear structuring elements (LSE) 
%   used in morphological filter elements on different scales. Either
%   linear or logarithmic scale interval is used.
%   INPUT     Num_scale  The number of scales used
%             D          The diameter range of breast masses
%             
%             type       The scale type (linear or logarithmic)
%   OUTPUT    len_bank    The magnitudes of the LSEs

if strcmp(type,'linear')
scale_interval = ceil((D(2) - D(1))/Num_scale);
len_bank = zeros(1,Num_scale+1);
    for l = 1:Num_scale+1
        len_bank (l) = D(1) + (l-1) * scale_interval;
    end
    % This is a linear bank
    len_bank(Num_scale+1) = D(2);

end


if strcmp (type, 'exponential')
   scale_interval =  (D(2)/D(1))^(1/Num_scale);
   for l = 1:Num_scale+1
        len_bank (l) = round( D(1) * (scale_interval^(l-1)) );
    end
    % This is a linear bank
    len_bank(Num_scale+1) = D(2);
    
end
end


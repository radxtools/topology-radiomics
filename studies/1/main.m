%%%Written by Marwa Ismail--April 30 2019
%%%%%%%Run local shape feature software%%%%%%%%%%%%%%%%%
close all;
clear all;

studies={'1'};
num_studies=size(studies,2);

for j=1:num_studies 
    Study_number=j;
    s=studies(1,j);
   new_folder = char(s);

Lesion = nii2mat('Lesion.nii');
Lesion(Lesion>0)=255;
[measures_Lesion,S_E]=SurfacePts(Lesion, [1 1 1]);
if(measures_Lesion~=0)

save('measures','measures_Lesion')

end



end



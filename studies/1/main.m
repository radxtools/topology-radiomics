%%%Written by Marwa Ismail--April 30 2019
%%%%%%%Run local shape feature software%%%%%%%%%%%%%%%%%
close all;
clear all;

Lesion = nii2mat('Lesion.nii');
Lesion(Lesion>0)=255;
[measures_Lesion,S_E]=SurfacePts(Lesion, [1 1 1]);
if(measures_Lesion~=0)
    save('measures','measures_Lesion')
end
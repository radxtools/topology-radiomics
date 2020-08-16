% Load *.nii subject and return the image volume

function img_volume = nii2mat(nii_path)

strcat(pwd,'\Segmentation\Code\M Files\load_nii');
if (exist(strcat(pwd,'\Segmentation\Code\M Files\load_nii'),'dir'))
    addpath(strcat(pwd,'\Segmentation\Code\M Files\load_nii'));
else
    fprintf(' - nii loading files not found!\n');
end

nii_volume = load_untouch_nii(nii_path);

img_volume = nii_volume.img;

img_volume = permute(img_volume,[2 1 3]);

for i = 1:size(img_volume,3)
    img_volume(:,:,i) = fliplr(img_volume(:,:,i));
end
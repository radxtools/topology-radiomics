files = dir('data/*.nii')
for i=2#:length(files)
    file_name = files(i).name
    printf("Processing %s \n",file_name);
    source_file_name = strcat('data/', file_name);
    data = nii2mat(source_file_name);
    destination_file_name = substr(file_name,1,-4);
    destination_file = strcat('data-m/', destination_file_name, ".mat");
    save('-6',destination_file,"data");
    [counts,bins] = hist(data(:)) 
end
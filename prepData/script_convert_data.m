% script_convert2nifty


flag_plot = false;

voxsize_in = 0.8;
voxsize_out = 0.4;


normFolder = fullfile(getenv('HOME'),'/Scratch/data/mrusv2/normalised');
h5fn_image_us = fullfile(normFolder,'us_images_resampled800.h5');
h5fn_image_mr = fullfile(normFolder,'mr_images_resampled800.h5');
h5fn_label_us = fullfile(normFolder,'us_labels_resampled800_post3.h5');
h5fn_label_mr = fullfile(normFolder,'mr_labels_resampled800_post3.h5');

outFolder = fullfile(getenv('HOME'),'/Scratch/data/mrusv2/demo');
mkdir(outFolder)
folder_us_image = fullfile(outFolder,'us_images');
folder_mr_image = fullfile(outFolder,'mr_images');
folder_us_label = fullfile(outFolder,'us_labels');
folder_mr_label = fullfile(outFolder,'mr_labels');
mkdir(folder_us_image)
mkdir(folder_mr_image)
mkdir(folder_us_label)
mkdir(folder_mr_label)

NumLandmarks_us = h5read(h5fn_label_us,'/num_labels');
NumLandmarks_mr = h5read(h5fn_label_mr,'/num_labels');

addpath ../../igitk/external/nifti/
nn = 0;
for n = 1:4:108
    
    GroupName = sprintf('/case%06d',n-1);
    
    image_us = anonymisingFilter(h5read(h5fn_image_us,GroupName), voxsize_out/voxsize_in, true);
    image_mr = anonymisingFilter(h5read(h5fn_image_mr,GroupName), voxsize_out/voxsize_in, true);
    if flag_plot
        figure, GrayMontage(image_us)
        figure, GrayMontage(image_mr)
        pause; close all
    end
    
    nn = nn+1;
    GroupName_sample = sprintf('case%06d',nn-1);
    save_nii(make_nii(image_us,voxsize_out*[1,1,1]), fullfile(folder_us_image,[GroupName_sample,'.nii']));
    save_nii(make_nii(image_mr,voxsize_out*[1,1,1]), fullfile(folder_mr_image,[GroupName_sample,'.nii']));
    
    num_labels = (floor(rand*3)+1);
    label_us = [];
    label_mr = [];
    for m = 1:num_labels
        BinName = sprintf('/case%06d_bin%03d',n-1,m-1);
        label_us(:,:,:,m) = anonymisingFilter(h5read(h5fn_label_us,BinName), voxsize_out/voxsize_in, false, true);
        label_mr(:,:,:,m) = anonymisingFilter(h5read(h5fn_label_mr,BinName), voxsize_out/voxsize_in, false, true);
        % check
        if nnz(label_us(:,:,:,m))==0 || nnz(label_us(:,:,:,m))==0, error('empty label(s)'); end
        if flag_plot
            figure, GrayMontage(label_us(:,:,:,m))
            figure, GrayMontage(label_mr(:,:,:,m))
            pause; close all
        end
        
    end
    BinName_sample = sprintf('case%06d_bin%03d',nn-1,m-1);
    save_nii(make_nii(label_us,voxsize_out*[1,1,1]), fullfile(folder_us_label,[GroupName_sample,'.nii']));
    save_nii(make_nii(label_mr,voxsize_out*[1,1,1]), fullfile(folder_mr_label,[GroupName_sample,'.nii']));
    
end

% compress
gzip(fullfile(folder_us_image,'*.nii'));
gzip(fullfile(folder_mr_image,'*.nii'));
gzip(fullfile(folder_us_label,'*.nii'));
gzip(fullfile(folder_mr_label,'*.nii'));

delete(fullfile(folder_us_image,'*.nii'));
delete(fullfile(folder_mr_image,'*.nii'));
delete(fullfile(folder_us_label,'*.nii'));
delete(fullfile(folder_mr_label,'*.nii'));

if ~exist('data', 'dir')
    mkdir('data');
end
for i = 0:299
    % TODO ignore existing like the python implementation
    case_dir = fullfile('data', sprintf('case_%05d', i));
    if ~exist(case_dir, 'dir')
        mkdir(case_dir);
    end
    websave(fullfile(case_dir, 'imaging.nii.gz'), sprintf('https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_%05d.nii.gz', i), 'timeout',20);
end
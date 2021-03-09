if ~exist('data', 'dir')
    mkdir('data');
end
for i = 0:299
    % TODO ignore existing like the python implementation
    case_dir = fullfile('data', sprintf('case_%05d', i));
    if ~exist(case_dir, 'dir')
        mkdir(case_dir);
    end
    websave(fullfile(case_dir, 'imaging.nii.gz'), sprintf('https://kits19.sfo2.digitaloceanspaces.com/master_%05d.nii.gz', i), 'timeout',20);
end
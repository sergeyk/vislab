try
    path(path, 'third-party/vlfeat/toolbox');
    path(path, 'third-party/llc');
    vl_setup;

    imsizes = [ 1 0.5 0.25 ];
    binSizes = [4 4 4];
    steps = ceil(step * imsizes);

    c = load(codebook);

    pyramid=[1, 3];

    files = dir(dirname);
    files = files(~[files.isdir]);

    betas = [];
    ids = cell(numel(files),1);
    for i = 1:numel(files)
        filename = files(i).name;
        k = findstr(filename,'.');
        id=filename(1:k(end-1)-1);

        fprintf('extracting feature for %s\n', id);

        I = im2single(imread(sprintf('%s/%s', dirname, filename)));
        beta = im2llc(I, imsizes, binSizes, steps, c.codebook, pyramid, knn, []);

        if isempty(betas)
            betas = sparse(size(beta,1), numel(files));
        end

        betas(:,i) = beta;
        ids{i} = id;
    end

    save(output, 'betas', 'ids', '-v7.3');
catch ME
    ME
    ME.message
    ME.stack
    ME.stack.file
    ME.stack.name
    ME.stack.line

    exit
end

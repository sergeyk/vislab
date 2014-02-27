function [model, cv_models, cv_pred, idx_internal, internal_models, internal_cv_models, internal_cv_pred] = liblinear_cv_train(labels, betas, nf, opt_str, pred_prob , seed, synsets)

s = RandStream('mt19937ar','Seed', seed);

n = numel(labels);

p = randperm(s,n);

fs = ceil(n/nf);
nf = ceil(n/fs);

cv_models = cell(nf,1);

internal_cv_models = cell(nf,1);

ts_opt_str = sprintf('-b %d', pred_prob);

cv_pred = zeros(n,numel(unique(labels)));

L = numel(synsets);
vec_label = labels_to_vec(labels, L);
full_vec_label = get_all_probs(vec_label,synsets);

heights = [synsets.height];
max_height = max(heights);

idx_internal = find(heights>0 & heights < max_height); %exclude root
num_internal = numel(idx_internal);

internal_cv_pred = zeros(n, num_internal);

for i = 1:nf
    a = fs*(i-1) + 1;
    b = fs* i;

    ts_idx = p(a:min(b,n));
    
    tr_mask = true(1,n);
    tr_mask(ts_idx) = 0;
    tr_idx = find(tr_mask);

    numel(tr_idx)
    numel(ts_idx)
    intersect(tr_idx, ts_idx)

    mymodel = train(labels(tr_idx), betas(:,tr_idx), opt_str, 'col');
    cv_models{i} = mymodel;
    
    [pred_labels, acc, dec] = predict(labels(ts_idx), betas(:,ts_idx), mymodel, ts_opt_str, 'col');

    cv_pred(ts_idx,mymodel.Label) = dec;

    internal_cv_models{i} = cell(num_internal, 1);
    %also train internal nodes
    for j = 1:num_internal
        my_label = full_vec_label(:,idx_internal(j));
        my_internal_model = train(my_label(tr_idx), betas(:, tr_idx), opt_str, 'col');
        
        internal_cv_models{i}{j} = my_internal_model;
        
        [pred_labels, acc, dec] = predict(my_label(ts_idx), betas(:,ts_idx), my_internal_model, ts_opt_str, 'col');
        if my_internal_model.Label(1) < 1 
            dec = -dec;
        end
        internal_cv_pred(ts_idx, j) = dec;
    end
end

model = train(labels, betas, opt_str, 'col');

internal_models = cell(num_internal,1);

for j = 1:num_internal
    my_label = full_vec_label(:,idx_internal(j));
    internal_models{j} = train(my_label, betas, opt_str,'col');
end

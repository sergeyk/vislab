path(path,'third-party/liblinear-1.8/matlab');

subset_id=0 % the subset to train on 
C=100; % the C parameter of SVM

seed=0; %seed for random numbers
nf=10; %number of folds in cross validation for calibration

meta_location = 'ilsvrc65_meta.mat'; 
M = load(meta_location); 

train_gt_map = read_gt('ilsvrc65.train.gt');

feature_location = sprintf('../features/ilsvrc65.train.subset%d.llc.mat',subset_id);
F = load(feature_location);
        
labels = ids_to_labels(F.ids, train_gt_map, M.class2id_map);

model_path=sprintf('../models/ilsvrc65.subset%d.C%d.model.mat', subset_id, C);

liblinear_str = sprintf('-s 2 -c %f -e 0.001 -B -1', C) %liblinear parameters

[model, cv_models, cv_pred, idx_internal, internal_models, internal_cv_models, internal_cv_pred] = liblinear_cv_train(labels, F.betas, nf, liblinear_str, 0, seed, M.synsets);

save(model_path, 'model','cv_models','cv_pred','idx_internal','internal_models','internal_cv_models','internal_cv_pred','seed');







        

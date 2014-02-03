
function a = rocSal( salmap , mask )

%     ROC area agreement between saliency map (salmap) and fixations (mask) 
% == good measure of HOW WELL salmap 'predicts' fixations
%
% - mask is the same size as salmap and
%   contains number of fixations at each
%   map location ( 0,1,2,..etc. )
%
% - gives the ROC score of the following binary classification problem 
%
%   the set of trials is this
%      {each (fixation,location)} UNION
%      {each location on the map without a fixation}
%
%   a true positive occurs
%      when a (fixation,location) pair is above threshold
%   a true negative occurs
%      when a (no fixation,location) pair is below threshold
%   a false negative occurs
%      when a (fixation,location) pair is below threshold
%   a false positive occurs
%      when a (no fixation,location) pair is above threshold
%
%   ROC curve plots TPR against FPR
%    where TPR = TP / (TP+FN) = TP / (number of ground-truth trues)
%          FPR = FP / (FP+TN) = FP / (number of ground-truth falses)
%
%  so if out of 10 fixations, 9 occur at one location, which is the 
%  only above threshold location, the TPR is 90%, and the FPR is 0%
%

% limit to 256 unique values
salmap = mat2gray(salmap);
if ( strcmp(class(salmap),'double') )
    % salmap = uint8(salmap * 255);
    salmap = uint8(salmap * 50);
end
    
t = getIntelligentThresholds( salmap(:) );
Nt = length(t);
p = zeros(Nt,2);

% number of (fixation,location) trials
Ntrues = sum(mask(:)); 

% number of (no fixation,location) trials
falses = (mask==0);
Nfalses = sum( falses(:) );
if ( Nfalses == 0 ) Nfalses = 1e-6; end

for ti = 1 : Nt   
    
  T = t(ti);    
  shouldbefix = salmap >= T;    
  
  TPm = mask .* shouldbefix;
  TP = sum( TPm(:) );
  tpr = TP / Ntrues;
  
  FPm = (mask==0) .* shouldbefix;
  FP = sum( FPm(:) );  
  fpr = FP / Nfalses;
  
  p(ti,:) = [ tpr fpr ];

end

a = areaROC(p);

function corlocs = compute_corloc(dets, imdb, gtdata)
% compute corloc 
% A success loclization means the bounding box of the largest score has a IoU > 0.5 with any ground truth

imdb.images.set(imdb.images.set == 2) = 1;
trainIdx = find(imdb.images.set == 1);

n_imgs = numel(trainIdx);
n_classes = 20;
% 1 success, -1 failed, 0 no ground truth class
LocMat = zeros(n_imgs, n_classes);

for i = 1:numel(trainIdx)
    id = trainIdx(i);
    image_label = find(imdb.images.label(id,:) > 0);
    gtbndbox = load_gt_bndbox(gtdata(i).objects, imdb.classes.name);
    boxScore = dets.scores{i};
    for l = image_label(:)'
        % find best box
        [~, box_idx] = max(boxScore(l,:));
        best_box = dets.boxes{i}(box_idx, :);
        % ious to gt boxes of label l
        overlaps = boxoverlap(gtbndbox{l}, best_box);
        max_overlap = max(overlaps);
        if max_overlap >= 0.5
            LocMat(i, l) = 1;
        else
            LocMat(i, l) = -1;
        end
    end  
end
corlocs = zeros(1, n_classes);
for c = 1:n_classes
    n_cor = length(find(LocMat(:,c) == 1));
    n_im = length(find(LocMat(:,c) ~= 0));
    corlocs(c) = n_cor / n_im;
    
end

end

function gtbndbox = load_gt_bndbox(objects, cls_names)
n_obj = numel(objects);
gtbndbox = cell(1, numel(cls_names));
for o = 1:n_obj
    obj_label = strfind(cls_names, objects.class);
    obj_label = find(not(cellfun('isempty', obj_label)));
    gtbndbox{obj_label} = [gtbndbox{obj_label}; objects(o).bbox(2,1,4,3)];
end
end

function [img, adjusted_rois] = load_data(jpgPath, batch, imdb, conv5_w, conv5_h, opts)

img = imread(fullfile(jpgPath,imdb.images.name{batch}));
% pick size
imSize = imdb.images.size(batch,:);
factor = min(opts.scale/imSize(1),opts.scale/imSize(2));
height = floor(factor*imSize(1));
width = floor(factor*imSize(2));
img = single(img);
img = imresize(img,[height, width],'Method','bilinear');

bboxes = double(imdb.images.boxes{batch});
nBoxes = size(bboxes,1);
adjusted_rois = zeros(nBoxes,5, 'single');
  
hw = imdb.images.size(batch,:);
h = hw(1);
w = hw(2);
  
imsz = size(img);

% flip image and bboxes if required
if opts.flip
    img = img(:,end:-1:1,:);
    %ims{b} = im(:,end:-1:1,:);
%     bbox = bboxes{b};
%     bbox(:,[2,4]) = w + 1 - bbox(:,[4,2]);
%     bboxes{b} = bbox; 
    bboxes(:,[2,4]) = w + 1 - bboxes(:,[4,2]);
end
% scale box
tbbox = scale_box(bboxes,[h,w],imsz);
% adjust rois
adj_bbox = spm_response_boxes(tbbox', conv5_w,  conv5_h, opts.offset1, opts.offset2, opts.stride);
adjusted_rois(:,2:end) = single(adj_bbox');

% minus mean
img = bsxfun(@minus, img, opts.averageImage);

img = permute(img, [2,1,3]);
img = reshape(img, width, height, 3, 1);
% rmin, cmin, rmax, cmax to xmin, ymin, xmax,ymax
adjusted_rois = adjusted_rois(:,[1,3,2,5,4]);
adjusted_rois = adjusted_rois';
adjusted_rois = reshape(adjusted_rois, 1,1,5,nBoxes);
end


function boxOut = scale_box(boxIn,szIn,szOut)
  
  h = szIn(1);
  w = szIn(2);

  bxr = 0.5 * (boxIn(:,2)+boxIn(:,4)) / w;
  byr = 0.5 * (boxIn(:,1)+boxIn(:,3)) / h;
 
  bwr = (boxIn(:,4)-boxIn(:,2)+1) / w;
  bhr = (boxIn(:,3)-boxIn(:,1)+1) / h;
  
  % boxIn center in new coord
  byhat = (szOut(1) * byr);
  bxhat = (szOut(2) * bxr);
  
  % relative width, height
  bhhat = szOut(1) * bhr;
  bwhat = szOut(2) * bwr;
  
  % transformed boxIn
  boxOut = [max(1,round(byhat - 0.5 * bhhat)),...
    max(1,round(bxhat - 0.5 * bwhat)), ...
    min(szOut(1),round(byhat + 0.5 * bhhat)),...
    min(szOut(2),round(bxhat + 0.5 * bwhat))];
end

% this func is borrowed from WSDDN
function nboxes = spm_response_boxes(boxes, w, h, offset1, offset2, stride)
  o0 = offset1;
  o  = offset2;
  ss = stride;
  if numel(ss)==1
    ss(2) = ss(1);
  end

  nboxes = [ ...
    floor((boxes(1,:) - o0 + o) / ss(1) + 0.5);
    floor((boxes(2,:) - o0 + o) / ss(2) + 0.5);
    ceil((boxes(3,:) - o0 - o) / ss(1) - 0.5);
    ceil((boxes(4,:) - o0 - o) / ss(2) - 0.5)];

  function a = fix_invalid(a)
    inval = a(1,:) > a(2,:);
    a(1,inval) = floor((a(1,inval) + a(2,inval))./2);
    a(2,inval) = a(1,inval);
  end

  nboxes([1 3],:) = fix_invalid(nboxes([1 3],:));
  nboxes([2 4],:) = fix_invalid(nboxes([2 4],:));

  nboxes = [...
    min(h-2, max(nboxes(1,:), 0));
    min(w-2, max(nboxes(2,:), 0));
    min(h-1, max(nboxes(3,:), 0));
    min(w-1, max(nboxes(4,:), 0))];
end







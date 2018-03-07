%evaluate model 
clear;
addpath('py-faster-rcnn/caffe-fast-rcnn/matlab');
% % set parameters
jpgPath = 'data/VOCdevkit/VOC2007/JPEGImages';
imdb = load(fullfile('data', 'VOC07_imdb.mat'));
imageScales = [480,576,688,864,1200];
exp_name = 'VOC07_VGGF';
save_path = fullfile('exp',exp_name);
% VGG-F
netdef = fullfile('models',[exp_name '.prototxt']);
model = fullfile('models', [exp_name, '.caffemodel']);

net = caffe.Net(netdef, model, 'test');
caffe.set_mode_gpu();
caffe.set_device(0);
testIdx = find(imdb.images.set == 3);

opts.averageImage = single(reshape([102.9801, 115.9465, 122.7717], 1, 1, 3));
opts.offset1 = 18.0;
opts.offset2 = 9.5;
opts.stride = 16;

scores = cell(1,numel(testIdx));
boxes = imdb.images.boxes(testIdx);
names = imdb.images.name(testIdx);
start = tic ;
for t=1:numel(testIdx)
  batch = testIdx(t);  
  
  scoret = [];
  for s=1:numel(imageScales)
    for f=1:2 % add flips
      %inputs = getBatch(bopts, imdb, batch, opts.imageScales(s), f-1 );
      opts.flip = f - 1;
      opts.scale = imageScales(s);
      imSz = imdb.images.size(batch,:);
      factor = min(opts.scale/imSz(1),opts.scale/imSz(2));
      nheight = floor(factor*imSz(1));
      nwidth = floor(factor*imSz(2));
      nROIS = size(imdb.images.boxes{batch},1);
      % caffe matlab shape: W,H,C,N
      net.blobs('data').reshape([nwidth, nheight, 3, 1]);
      net.blobs('rois_tmp').reshape([1, 1, 5,nROIS]);
      net.reshape();
      conv5_shape = net.blobs('conv5').shape;
      conv5_w = conv5_shape(1); conv5_h = conv5_shape(2);
      [img, adjusted_rois] = load_data(jpgPath, batch, imdb, conv5_w, conv5_h, opts);

      net.forward({img, adjusted_rois});
      
      if isempty(scoret)
        %scoret = squeeze(gather(net.vars(detLayer).value));
        scoret = net.blobs('xTimes').get_data();
      else
        %scoret = scoret + squeeze(gather(net.vars(detLayer).value));
        scoret = scoret + net.blobs('xTimes').get_data();
      end
    end
  end
  scores{t} = scoret;
  %show speed
  time = toc(start) ;
  n = t * 2 * numel(imageScales) ; % number of images processed overall
  speed = n/time ;
  if mod(t,10)==0
    fprintf('test %d / %d speed %.1f Hz\n',t,numel(testIdx),speed);
  end
end

dets.names  = names;
dets.scores = scores;
dets.boxes  = boxes;
save(fullfile(save_path, [exp_name '_dets.mat']), '-struct', 'dets', '-v7.3');








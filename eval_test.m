% eval test
addpath('pascal');

dataDir = 'data';
exp_name = 'VOC07_VGGF';
dataset = 'VOC07';
save_path = fullfile('exp',exp_name);

dets = load(fullfile(save_path, [exp_name '_dets.mat']));

imdb = load(fullfile('data', 'VOC07_imdb.mat'));
testIdx = find(imdb.images.set == 3);

addpath(fullfile(dataDir,'VOCdevkit','VOCcode'));
VOCinit;
VOCopts.testset = 'test';
VOCopts.annopath = fullfile(dataDir,'VOCdevkit','VOC2007','Annotations','%s.xml');
VOCopts.imgsetpath = fullfile(dataDir,'VOCdevkit','VOC2007','ImageSets','Main','%s.txt');
VOCopts.localdir = fullfile(dataDir,'VOCdevkit','local','VOC2007');

cats = VOCopts.classes;
ovTh = 0.4;
scTh = 1e-3;
aps = zeros(numel(cats),1);
rfid = fopen(fullfile(save_path, 'det_aps.txt'), 'w');

for cls = 1:numel(cats)
  vocDets.confidence = [];
  vocDets.bbox       = [];
  vocDets.ids        = []; 
  for i=1:numel(dets.scores)
    
    scores = double(dets.scores{i}');
    boxes  = double(imdb.images.boxes{testIdx(i)});
    
    boxesSc = [boxes,scores(:,cls)];
    boxesSc = boxesSc(boxesSc(:,5)>scTh,:);
    pick = nms(boxesSc, ovTh);
    boxesSc = boxesSc(pick,:);
    
    vocDets.confidence = [vocDets.confidence;boxesSc(:,5)];
    vocDets.bbox = [vocDets.bbox;boxesSc(:,[2 1 4 3])];
    if strcmp(dataset, 'VOC07')
      vocDets.ids = [vocDets.ids; repmat({dets.names{i}(1:6)},size(boxesSc,1),1)];
    else
      [~,voc_id,~] = fileparts(dets.names{i});
      vocDets.ids = [vocDets.ids; repmat({voc_id},size(boxesSc,1),1)];
    end
  end 
  if strcmp(dataset, 'VOC10') || strcmp(dataset, 'VOC12')
      for j = 1:numel(vocDets.ids)
        fprintf(fid, '%s %f %f %f %f %f\n', vocDets.ids{j}, vocDets.confidence(j), vocDets.bbox(j,1), vocDets.bbox(j,2), vocDets.bbox(j,3), vocDets.bbox(j,4));
      end
      fclose(fid);
  end
  if(strcmp(dataset, 'VOC07'))
    [rec,prec,ap] = wsddnVOCevaldet(VOCopts,cats{cls},vocDets,0);
    fprintf('%s %.1f\n',cats{cls},100*ap);
    fprintf(rfid, '%s %.1f\n', cats{cls}, 100*ap);
    aps(cls) = ap;
  end
end
fprintf('mean %.1f\n',100*mean(aps));
fprintf(rfid, 'mean %.1f\n', 100*mean(aps));


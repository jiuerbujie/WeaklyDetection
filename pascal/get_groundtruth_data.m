%get ground truth data
addpath('../data/VOCdevkit/VOCcode');
clear;
tic;
VOCinit;
cats = VOCopts.classes;
opts.dataDir = '../data';
VOCopts.testset = 'trainval';
VOCopts.annopath = fullfile(opts.dataDir,'VOCdevkit','VOC2012','Annotations','%s.xml');
VOCopts.imgsetpath = fullfile(opts.dataDir,'VOCdevkit','VOC2012','ImageSets','Main','%s.txt');
%VOCopts.localdir = fullfile(opts.dataDir,'VOCdevkit','local','VOC2007');

VOCopts.annocachepath='voc12_%s_anno_cache.mat';
cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
if ~exist(cp, 'file')
    [gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    for i=1:length(gtids)
    % read annotation
    recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
    save(cp,'gtids','recs');
else
   load(cp); 
end


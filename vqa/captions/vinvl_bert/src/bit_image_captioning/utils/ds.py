
import base64
import numpy as np
import os.path as op
import json
import torch
from torch.utils.data import Dataset
import torch.distributed as dist


from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import (tsv_writer, concat_tsv_files,delete_tsv_files, reorder_tsv_keys)
from oscar.utils.misc import (mkdir, set_seed, load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,ScstRewardCriterion)
from oscar.utils.logger import setup_logger



from ..tokenizers.bert_tokenizer import CaptionTensorizer
from ..modeling.modeling_bert_utils import ConstraintFilter, ConstraintBoxesReader,FiniteStateMachineBuilder



class CaptionTSVDataset(Dataset):


    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, 
            is_train=True, mask_prob=0.15, max_masked_tokens=3, **kwargs):
        
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        self.caption_file = find_file_path_in_yaml(self.cfg.get('caption'), self.root)
        
        assert op.isfile(self.feat_file)
        if add_od_labels: assert op.isfile(self.label_file)
        if is_train: assert op.isfile(self.caption_file) and tokenizer is not None

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)
        self.captions = []
        if self.caption_file and op.isfile(self.caption_file):
            with open(self.caption_file, 'r') as f:
                self.captions = json.load(f)

        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                max_seq_length, max_seq_a_length, mask_prob, max_masked_tokens,
                is_train=is_train)
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.key2captions = self.prepare_image_key_to_captions()

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self):
        if self.captions:
            key2captions = {key: [] for key in self.image_keys}
            for cap in self.captions:
              # print(cap['image_id'])
              key2captions[cap['image_id']].append(cap['caption'])
            return key2captions

    def get_image_index(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            img_key = img_cap_pair['image_id']
            return self.key2index[img_key]
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        feat_info = json.loads(self.feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info['num_boxes']
        features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
                ).reshape((num_boxes, -1))
        #return torch.Tensor(features)
        return torch.Tensor(np.array(features))

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair['caption']
        return ""

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels = " ".join([l['class'] for l in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = op.splitext(self.caption_file)[0] + '_coco_format.json'
        return cap_file

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(img_idx)
        caption = self.get_caption(idx)
        od_labels = self.get_od_labels(img_idx)
        example = self.tensorizer.tensorize_example(caption, features, text_b=od_labels)
        return img_key, example

    def __len__(self):
        if self.is_train:
            return len(self.captions)
        return self.get_valid_tsv().num_rows()

# .........................................................................................................

class CaptionTSVDatasetWithConstraints(CaptionTSVDataset):
    """
    Providing inputs for inference with Constraint Beam Search

    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

    def __init__(
        self, yaml_file,
        nms_threshold=0.85,
        max_given_constraints=3, **kwargs
    ):
        super().__init__(yaml_file, **kwargs)
        boxes_tsvpath = find_file_path_in_yaml(self.cfg['cbs_box'], self.root)
        constraint2tokens_tsvpath = find_file_path_in_yaml(self.cfg['cbs_constraint'], self.root)
        tokenforms_tsvpath = find_file_path_in_yaml(self.cfg['cbs_tokenforms'], self.root)
        hierarchy_jsonpath = find_file_path_in_yaml(self.cfg['cbs_hierarchy'], self.root)

        self._boxes_reader = ConstraintBoxesReader(boxes_tsvpath)
        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(self.tokenizer,
                constraint2tokens_tsvpath, tokenforms_tsvpath,
                max_given_constraints)

    def __getitem__(self, index):
        img_key, example = super().__getitem__(index)

        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[img_key]

        candidates = self._constraint_filter(
            constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
        )
        num_constraints = len(candidates)
        fsm, nstates = self._fsm_builder.build(candidates)

        return img_key, example + (fsm, num_constraints, )


# .........................................................................................................

def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens)
    if args.use_cbs:
        dataset_class = CaptionTSVDatasetWithConstraints
    else:
        dataset_class = CaptionTSVDataset
    return dataset_class(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
            is_train=False)

# .........................................................................................................


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

# .........................................................................................................

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

# .........................................................................................................

def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True, is_valloss=False):
    global logger
    dataset = build_dataset(yaml_file, tokenizer, args, 
        is_train=(is_train and not args.scst))
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        if is_valloss:
            logger.info("Validate with {} images per GPU.".format(images_per_gpu))
            logger.info("Total batch size {}".format(images_per_batch))
            logger.info("Total validation steps {}".format(num_iters))
        else:
            logger.info("Train with {} images per GPU.".format(images_per_gpu))
            logger.info("Total batch size {}".format(images_per_batch))
            logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader

# .........................................................................................................


# {'metadata': {'image_id': 297147,
#   'question_id': 2971475,
#   'question_type': 'one',
#   'answer_type': 'other',
#   'confidence': 3},
#  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480>,
#  'question': {'en': 'What sport can you use this for?',
#   'ar': 'في أي رياضة يمكنك استخدام هذا؟'},
#  'answers': {'en': ['race',
#    'race',
#    'race',
#    'race',
#    'race',
#    'race',
#    'motocross',
#    'motocross',
#    'ride',
#    'ride'],
#   'ar': ['سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'موتوكروس',
#    'موتوكروس',
#    'يركب',
#    'يركب'],
#   'raw_en': ['racing',
#    'racing',
#    'racing',
#    'racing',
#    'racing',
#    'racing',
#    'motocross',
#    'motocross',
#    'riding',
#    'riding'],
#   'raw_ar': ['سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'سباق',
#    'موتوكروس',
#    'موتوكروس',
#    'يركب',
#    'يركب'],
#   'confidence': ['yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes',
#    'yes'],
#   'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}
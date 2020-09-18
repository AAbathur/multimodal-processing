import _pickle as cPickle
import torch
import torch.utils.data as data
import config
import json
import re
import utils
import h5py
import numpy as np


preloaded_vocab = None

def get_loader(train=False, val=False, test=False, trainval=False):
    """ Returns a data loader for the desired split """
    split = VQA(
        utils.path_for(train=train, val=val, test=test, trainval=trainval, question=True),
        utils.path_for(train=train, val=val, test=test, trainval=trainval, answer=True),
        config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
        answerable_only=train or trainval,
        dummy_answers=test,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train or trainval,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare_questions(questions_json):
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        question = _special_chars.sub('', question)
        yield question.split(' ')

def prepare_answers(answers_json):
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    
    def process_punctuation(s):
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()
    
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))

class VQA(data.Dataset):
    def __init__(self, questions_path, answers_path, image_features_path, answerable_only=False, dummy_answers=False):
        super(VQA, self).__init__()
        with open(questions_path, 'r') as fq:
            questions_json = json.load(fq)
        with open(answers_path, 'r') as fa:
            answers_json = json.load(fa)
        
        if preloaded_vocab:
            vocab_json = preloaded_vocab
        else:
            with open(config.vocabulary_path, 'r') as fv:
                vocab_json = json.load(fv)
            word2idx, idx2word = cPickle.load(open(config.glove_index, 'rb'))
            vocab_json['question'] = word2idx
        self.question_ids = [q['question_id'] for q in questions_json['questions']]
        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']
        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        # self.question[i] : question[i] len(question[i])
        self.questions = [self._encoder_question(q) for q in self.questions]
        self.answers = [self._encoder_answers(a) for a in self.answers]
        # v
        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for q in questions_json['questions']]

        self.dummy_answers = dummy_answers

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable(not self.answerable_only)

    @property
    def num_tokens(self):
        return len(self.token_to_index)

    @property  
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            data_max_length = max(map(len, self.questions))
            self._max_length = min(config.max_q_length, data_max_length)
        return self._max_length
    
    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self, count=False):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        if count:
            number_indices = torch.LongTensor([self.answer_to_index[str(i)] for i in range(0,8)])
        for i, answer in enumerate(self.answers):
            if count:
                answers = answers[number_indices]
            answer_has_index = len(answer.nonzero()) > 0
            if answer_has_index:
                answerable.append(i)
        return answerable
        
    def _encoder_question(self, question):
        """ turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long().fill_(self.num_tokens)
        for i, token in enumerate(question):
            if i >= self.max_question_length:
                break
            index = self.token_to_index.get(token, self.num_tokens - 1)
            vec[i] = index
        return vec, min(len(question), self.max_question_length)
    
    def _encoder_answers(self, answers):
        """ turn an answer into a vector """
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _load_image(self, image_id):
        if not hasattr(self, 'feature_file'):
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_index[image_id]
        img = self.features_file['features'][index]
        boxes = self.features_file['boxes'][index]
        widths = self.features_file['widths'][index]
        heights = self.features_file['heights'][index]
        obj_mask = (img.sum(0) > 0).astype(int)
        return torch.from_numpy(img).transpose(0,1), torch.from_numpy(boxes).transpose(0,1), torch.from_numpy(obj_mask), widths, heights
    
    def __getitem__(self, item):
        if self.answerable_only:
            item = self.answerable[item]
        q, q_length = self.questions[item]
        q_mask = torch.from_numpy((np.arange(self.max_question_length) < q_length).astype(int))
        if not self.dummy_answers :
            a = self.answers[item]
        else:
            a = 0
        image_id = self.coco_ids[item]
        v, b, obj_mask, width, height = self._load_image(image_id)
        if config.normalize_box:
            assert b.shape[1] == 4
            b[:, 0] = b[:, 0] / float(width)
            b[:, 1] = b[:, 1] / float(height)
            b[:, 2] = b[:, 2] / float(width)
            b[:, 3] = b[:, 3] / float(height)
        return v, q, a ,b, item, obj_mask.float(), q_mask.float(), q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)
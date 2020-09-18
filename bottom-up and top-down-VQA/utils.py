import os
import json

import torch
import config

def path_for(train=False, val=False, test=False, question=False, trainval=False, answer=False):
    assert train + val + test + trainval == 1
    assert question + answer == 1

    if train:
        split = 'train2014'
    elif val:
        split = 'val2014'
    elif trainval:
        split = 'trainval2014'
    else:
        split = config.test_split
    
    if question:
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        if test:
            split = 'val2014'
        fmt = 'v2_{1}_{2}_annotations.json'
    s = fmt.format(config.task, config.dataset, split)
    return os.path.join(config.qa_path, s)
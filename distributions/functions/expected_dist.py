# if space_group < 30 then 30 for all
# overall/9

import numpy as np
import json

def predict_dist(overall):
    '''
    parameters
    ----------
    overall: dictionary
    overall amount of space groups in all datasets
    
    returns
    -------
    '''
    range = np.arange(1,231)
    train_redist = {}
    test_redist = {}
    dev_redist = {}
    all_redist = {}
    for x in range:
        sg = overall['Space Group {}'.format(x)]
        if sg < 90:
            all_redist['Space Group {}'.format(x)] = 30
            train_redist['Space Group {}'.format(x)] = 0
            test_redist['Space Group {}'.format(x)] = 0
            dev_redist['Space Group {}'.format(x)] = 0
        else:
            y = int(sg/9)
            all_redist['Space Group {}'.format(x)] = 0
            train_redist['Space Group {}'.format(x)] = y*7
            test_redist['Space Group {}'.format(x)] = y
            dev_redist['Space Group {}'.format(x)] = y
            
    with open('train_redist.json', 'w') as tr:
        json.dump(train_redist, tr)
        
    with open('test_redist.json', 'w') as te:
        json.dump(test_redist, te)
        
    with open('dev_redist.json', 'w') as de:
        json.dump(dev_redist, de)
    
    with open('all_redist.json', 'w') as a:
        json.dump(all_redist, a)
        
if __name__ == '__main__':
    json_path = 'overall_distribution.json'

    with open(json_path, 'r') as f:
        data = json.load(f)
        nums = list(range(1,231))
        data = np.array([data['Space Group {}'.format(num)] for num in nums])
    
    keys = np.arange(1, 231, dtype=int)
    overall_dict = {}
    for key,val in zip(keys,data):
        overall_dict['Space Group {}'.format(key)] = val.item()
    print('Dictionary created.')

    predict_dist(overall_dict)

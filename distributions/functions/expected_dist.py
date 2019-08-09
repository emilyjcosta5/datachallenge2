# if space_group < 30 then 30 for all
# overall/9

import numpy as np

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
    for x in range:
        sg = overall['Space Group {}'.format(x)]
        if sg < 90:
            train_redist['Space Group {}'.format(x)] = 30
            test_redist['Space Group {}'.format(x)] = 30
            dev_redist['Space Group {}'.format(x)] = 30
        else:
            y = int(sg/9)
            train_redist['Space Group {}'.format(x)] = y*7
            test_redist['Space Group {}'.format(x)] = y
            dev_redist['Space Group {}'.format(x)] = y
            
    with open('train_redist.json', 'w') as tr:
        json.dump(train_redist, tr)
        
    with open('test_redist.json', 'w') as te:
        json.dump(test_redist, te)
        
    with open('dev_redist.json', 'w') as de:
        json.dump(dev_redist, de)

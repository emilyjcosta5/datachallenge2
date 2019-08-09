from functions.expected_dist import *

json_path = 'dataframes/overall_distribution.json'

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
print('done')

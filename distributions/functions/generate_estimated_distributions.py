# yuh

from pandas import DataFrame
import json
import numpy as np
import random

def _convert_JSON_to_arr(JSON_path):
    with open(JSON_path, "r") as JSON_file:
        data = json.load(JSON_file)
        space_group_inds = list(range(1, 231))
        data = np.array([data['Space Group {}'.format(ind)] for ind in space_group_inds])

    return data

if __name__ == '__main__':
    jsonPath = "../dataframes/overall_distribution.json"
    space_group_distribution = _convert_JSON_to_arr(jsonPath)

    dictTrain = {}
    dictDev = {}
    dictTest = {}
    dicts = [dictTrain, dictDev, dictTest]
    saveNames = ["../dataframes/estDistTrain.json", "../dataframes/estDistDev.json", "../dataframes/estDistTest.json"]
    ratios = [7, 1, 1]
    dictList = []

    for i in range(len(dicts)):
        for j in range(ratios[i]):
            dictList.append(dicts[i])

    print("dictList is now {} elements long".format(len(dictList)))

    for k in range(space_group_distribution.size):
        if space_group_distribution[k] < 30:
            for aDict in dicts:
                aDict["Space Group {}".format(k+1)] = str(space_group_distribution[k])
        else:
            for l in range(space_group_distribution[k]):
                randNum = random.randrange(len(dictList))
                try:
                    dictList[randNum]["Space Group {}".format(k+1)] = str(int(dictList[randNum]["Space Group {}".format(k+1)]) + 1)
                except KeyError:
                    dictList[randNum]["Space Group {}".format(k+1)] = str(1)

    for m in range(len(saveNames)):
        with open(saveNames[m], "w") as saveFile:
            json.dump(dicts[m], saveFile)



import json
import csv

def getJSonObject():
    essayStringList = ''
    with open('../data/essay_corpus.json', errors='ignore') as f:
        for jsonObj in f:
            essayStringList = essayStringList + jsonObj
    jsonEssayObject = json.loads(essayStringList)
    return jsonEssayObject

trainIds=[]
testIds=[]
traindata=[]
testdata=[]
JsonObject=getJSonObject()

def splitdata():
    global trainIds,testIds,JsonObject,traindata,testdata
    with open('../data/train-test-split.csv', encoding="latin-1") as csvfile:
        spamreader = list(csv.reader(csvfile, delimiter=";"))
    for row in spamreader:
        if row[1] == "TRAIN":
            essayid = int(str(row[0]).replace("essay", ""))
            trainIds.append(essayid)
        elif row[1] == "TEST":
            essayid = int(str(row[0]).replace("essay", ""))
            testIds.append(essayid)
    for jsondata in JsonObject:
        if trainIds.__contains__(int(jsondata['id'])):
            traindata.append(jsondata)
        elif testIds.__contains__(int(jsondata['id'])):
            testdata.append(jsondata)

splitdata()
with open('../data/traindata.json', 'w',encoding="latin-1") as f:
    json.dump(traindata, f,indent=4)

with open('../data/testdata.json', 'w',encoding="latin-1") as f:
    json.dump(testdata, f,indent=4)




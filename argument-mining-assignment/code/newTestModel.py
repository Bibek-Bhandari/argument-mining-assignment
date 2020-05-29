import joblib
from sklearn.metrics import f1_score

clf=joblib.load('my_dumped_classifier.pkl')
feature_extraction=joblib.load('feature_extraction.pkl')

test_X = []
test_Y = []
# 1. iterate over file line-by-line
# 2. strip line of newline symbols
# 3. split line by spaces into list (of number strings)
# 4. convert number substrings to int values
# 5. convert map object to list
with open('data/traindata.txt') as fp:
	for line in fp:
		print(line)
		if (len(line.strip().split('\t')) > 1):
			test_X.append(line.strip().split('\t')[0])
			test_Y.append(line.strip().split('\t')[1])
		else:
			print(line)


test_X = test_X[25000:]
test_Y = test_Y[25000:]





test_X_vector = feature_extraction.transform(test_X)



predictions = clf.predict(test_X_vector)
print(len(predictions))
print(len(test_Y))
with open('data/testpred.txt', 'w') as f:
	for i in range(len(test_X)):
		f.write(test_X[i]+"\t"+predictions[i]+"\n")

gt_bio   = [x.split('\t') for x in open('data/testpred.txt').readlines() if x !='\n']
pred_bio = [x.split('\t') for x in open('data/testdata.txt').readlines() if x !='\n']

pred_bio1=[]
with open('data/testpred.txt') as fp:
	for line in fp:
		if (len(line.strip().split('\t')) > 1):
			pred_bio1.append(line.strip().split('\t')[1])

gt_bio_labels=[]
pt_bio_labels=[]

for x in gt_bio:
	gt_bio_labels.append(x[1])
for x in pred_bio:
	pt_bio_labels.append(x[1])

print(f1_score(test_Y, pred_bio1, average='macro'))
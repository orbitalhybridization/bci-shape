# Classification from: https://www.baeldung.com/cs/svm-multiclass-classification
# Confusion matrix plotting from: https://www.machinecurve.com/index.php/2020/05/05/how-to-create-a-confusion-matrix-with-scikit-learn/

# Imports
import scipy as sci
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.multiclass import OneVsRestClassifier as ovr
import sklearn.model_selection as model_selection

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import copy


"""
files = [("session_data/abigail_psd_data_normalized.m","session_data/abigail_targets.m",12,"abigail"),
			("session_data/m_psd_data_normalized.m","session_data/m_targets.m",30,"m"),
			("session_data/maxb_psd_data_normalized.m","session_data/maxb_targets.m",10,"maxb"),
			("session_data/azure_psd_data_normalized.m","session_data/azure_targets.m",10,"azure"),
			("session_data/maxh_psd_data_normalized.m","session_data/maxh_targets.m",10,"maxh"),
			("session_data/seabass_psd_data_normalized.m","session_data/seabass_targets.m",13,"seabass")]
"""

files = [("session_data/clay_psd_data_normalized.m","session_data/clay_targets.m",30,"clay")]
# Set up for metrics
total_average = 0
max_classification = 0
classes = ['Cube','Sphere','Pyramid','Cone','Cylinder']
true_vs_pred_summary = {}
one_vs_all_summary = {}
for true_class in classes:
	one_vs_all_summary[true_class] = 0
	preds = {}
	for pred in classes:
		preds[pred] = 0
	true_vs_pred_summary[true_class] = [0,preds]


for shapefile,targetfile,num_features,ID in files:

	print "\n\n********************************\n"+ID+"\n********************************\n"

	### File I/O + Setup
	shapes = sci.io.loadmat(shapefile)['psd_data_norm']
	shapes = shapes[0:50,:,3:5] # reshape to the last two frequency bands
	targets = sci.io.loadmat(targetfile,chars_as_strings=True)['targets'][0]

	# Parse targets and reshape shapes
	#targets = [targets[index][0].encode('ascii') for index in range(len(targets))]
	targets = [targets[index][0].encode('ascii') for index in range(50)]
	shapes = sci.reshape(shapes,(50,122))
	classes = ['Cube','Sphere','Pyramid','Cone','Cylinder']

	### Feature Selection (Top 20 Features using chi-squared as scoring)
	features_selected = SelectKBest(chi2, k=num_features).fit_transform(shapes,targets)

	### Model setup

	k = 5
	kf = model_selection.KFold(n_splits=k, random_state=None)
	# Set up model
	#shapes_train,shapes_test,targets_train,targets_test = model_selection.train_test_split(features_selected,targets,train_size=0.80,test_size=0.20,random_state=None)

	# Create LDA classifier
	#lda_classifier = lda(solver='svd',shrinkage=None,priors=None,n_components=None,store_covariance=False,tol=0.0001).fit(shapes_train,targets_train)
	lda_classifier = lda(solver='svd',shrinkage=None,priors=None,n_components=None,store_covariance=False,tol=0.0001)

	### Prediction & Evaluation
	
	acc_score = []
	for train_index , test_index in kf.split(features_selected):
		shapes_train , shapes_test = features_selected[train_index,:],features_selected[test_index,:]
		targets_train , targets_test = [targets[t] for t in train_index] , [targets[t] for t in test_index]
		print targets_test
		lda_classifier.fit(shapes_train,targets_train)
		pred_lda = lda_classifier.predict(shapes_test)
		
		acc = accuracy_score(pred_lda , targets_test)
		acc_score.append(acc)
		 
	accuracy_lda = sum(acc_score)/k
	

	# LDA
	#pred_lda = lda_classifier.predict(shapes_test)
	#accuracy_lda = accuracy_score(targets_test,pred_lda)
	#print targets_test
	print('Accuracy LDA: %.2f' % (accuracy_lda*100))
	# do some metrics
	total_average += (accuracy_lda*100)
	if (accuracy_lda*100) > max_classification: max_classification = accuracy_lda*100

	for index in range(len(shapes_test)):

		true_vs_pred_summary[targets_test[index]][1][pred_lda[index]] += 1
		true_vs_pred_summary[targets_test[index]][0] += 1

	# One vs. all classifier, prediction, & evaluation
	for class_name in classes:

		# These loops rename the targets to one vs. all
		rename_targets_train = copy.deepcopy(targets_train)
		rename_targets_test = copy.deepcopy(targets_test)
		for index in range(len(targets_train)):
			if targets_train[index] != class_name:
				rename_targets_train[index] = 'Other'
		for index in range(len(targets_test)):
			if targets_test[index] != class_name:
				rename_targets_test[index] = 'Other'

		# Now we can train a classifier on these new targets
		one_vs_all_classifier = lda(solver='svd',shrinkage=None,priors=None,n_components=None,store_covariance=False,tol=0.0001).fit(shapes_train,rename_targets_train)
		pred_lda = one_vs_all_classifier.predict(shapes_test)
		accuracy_lda = accuracy_score(rename_targets_test,pred_lda)
		one_vs_all_summary[class_name] += (accuracy_lda*100)
		print('Accuracy One vs. All (' + class_name + '): %.2f' % (accuracy_lda*100))


print "\n\nMetrics:\n"
total_average = float(total_average) / float(len(files))

#print sum(true_vs_pred_summary['Cone'].values())

print "Total Classification Average: ",total_average
print "Max Classification: "+str(max_classification)

for entry in one_vs_all_summary:
	one_vs_all_summary[entry] = float(one_vs_all_summary[entry]) / float(len(files))
	print('Average Accuracy One vs. All (' + entry + '): %.2f' % one_vs_all_summary[entry])

for true_class in true_vs_pred_summary:
	for prediction in true_vs_pred_summary[true_class][1]:
		true_vs_pred_summary[true_class][1][prediction] = (float(true_vs_pred_summary[true_class][1][prediction]) / float(true_vs_pred_summary[true_class][0]))*100

print "\nTrue vs. Pred: ",true_vs_pred_summary
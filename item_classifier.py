import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics, svm, grid_search, ensemble, neighbors, naive_bayes


# to avoid truncating stuff
pd.set_option('display.max_colwidth', 800)
pd.set_option('display.max_rows', 1000)


def main():

	datasets = prepDatasets('data/training_data.csv')
	exploreClassifiers(datasets, 'priority_sblc_wpusa', 'match_id')




'''
prepDatasets
============
Given the path to a csv file with training observations,
extract features from the CSV to build a df suitable for machine learning.
'''
def prepDatasets(csv_filepath):
	info_df = pd.read_csv(csv_filepath)

	# recode priority columns as dummy vars
	info_df['priority_wpusa_high'] = np.where(info_df['priority_wpusa']=='yellow', 1, 0)
	info_df['priority_wpusa_low'] = np.where(info_df['priority_wpusa']=='blue', 1, 0)
	info_df['priority_wpusa_any'] = info_df['priority_wpusa_high'] + info_df['priority_wpusa_low']

	info_df['priority_sblc_high'] = np.where(info_df['priority_sblc']=='yellow', 1, 0)
	info_df['priority_sblc_low'] = np.where(info_df['priority_sblc']=='blue', 1, 0)
	info_df['priority_sblc_any'] = info_df['priority_sblc_high'] + info_df['priority_sblc_low']

	info_df['priority_sblc_wpusa'] = np.where((info_df['priority_wpusa_any']==1) | (info_df['priority_sblc_any']==1), 1, 0)

	# code agency type
	with open('../agenda-parser/agencies_list.json') as data_file:
		agencies_list = json.load(data_file)
	agency_types = {agency['agency_id']: agency['agency_type'] for agency in agencies_list}
	info_df['agency_type'] = info_df['agency'].map(agency_types)

	# remove any lines where item text is NaN
	info_df = info_df.dropna(subset = ['item_text'])
	# redefine the index to fix merge issues with the dtm
	info_df.index = range(0,len(info_df))

	# # remove unwanted columns - TRY MAKING SOME OF THESE INTO FEATURES LATER
	# cols_to_drop = ['priority_sblc', 'priority_ibew', 'priority_unite', 'meeting_date', 'item_text_raw', 'meeting_sections.section_name', 'meeting_sections.section_number', 'item_type', 'item_recommendation', 'item_number', 'item_details','boarddocs_id', 'agency', 'priority_wpusa', 'item_text', 'priority_wpusa_high', 'priority_wpusa_low', 'agency_type']
	# features_df = info_df.drop(cols_to_drop, axis=1)

	# # merge with the DTM
	# features_df = pd.concat([features_df, dtm], axis=1)

	# set the match id to be the index of the info df
	info_df.set_index('match_id', inplace=True, drop=False)

	datasets = {'info_df': info_df}
	dtm_parts = buildDTM(info_df, 'item_text')
	datasets['dtm'] = dtm_parts[0]
	datasets['dtm_terms'] = dtm_parts[1]

	return datasets



'''
buildDTM
========
Given a dataframe and a column name, builds a document-term matrix
from the text of that column.
Returns the DTM as a pandas DF.
'''
def buildDTM(df, colname):
	text_list = df[colname].tolist()

	vectorizer = CountVectorizer(strip_accents="ascii", ngram_range=(1,3), stop_words='english', max_df=0.9, min_df=0.001)
	counts_matrix = vectorizer.fit_transform(text_list)
	
	return [counts_matrix, vectorizer.get_feature_names()]



'''
exploreClassifiers
==================
Explore which items different classifiers are catching,
and try combining the models.
'''
def exploreClassifiers(datasets, y_colname, id_colname):

	X_train, X_test, y_train, y_test, ids_train, ids_test = splitObs(datasets['dtm'], datasets['info_df'], y_colname, id_colname)
	test_train_sets = {'X_train': X_train, 
					'X_test': X_test, 
					'y_train': y_train, 
					'y_test': y_test, 
					'ids_train': ids_train, 
					'ids_test': ids_test}

	preds_lr = classifyLogisticRegression(datasets, test_train_sets, y_colname, id_colname)
	preds_svm = classifySVM(datasets, test_train_sets, y_colname, id_colname)
	preds_nb = classifyNaiveBayes(datasets, test_train_sets, y_colname, id_colname)
	preds_knn = classifyNearestNeighbors(datasets, test_train_sets, y_colname, id_colname)
	# # classifyRandomForest(datasets, info_df)

	pred_classes = {'lr': preds_lr[0], 'svm': preds_svm[0], 'nb': preds_nb[0], 'knn': preds_knn[0]}
	pred_probs_train = {'lr': preds_lr[1][:,1], 'svm': preds_svm[1][:,1], 'nb': preds_nb[1][:,1], 'knn': preds_knn[1][:,1]}
	pred_probs_test = {'lr': preds_lr[2][:,1], 'svm': preds_svm[2][:,1], 'nb': preds_nb[2][:,1], 'knn': preds_knn[2][:,1]}

	# compare the results of each classifier
	compareClassifiers(datasets, test_train_sets, pred_classes)

	# try combining the predicted probabilities from each classifier into an ensemble classifier
	combineClassifiers(datasets, test_train_sets, pred_probs_train, pred_probs_test)


'''
splitObs
========
Given a matrix of features, a dataframe, a y column name, and an id column name,
splits the dataset randomly into training and testing sets.
Returns 6 arrays: X_train, X_test, y_train, y_test, ids_train, ids_test
'''
def splitObs(features, df, y_colname, id_colname):
	y = np.array(df[y_colname])
	ids = np.array(df[id_colname])
	return train_test_split(features, y, ids, test_size=0.33, stratify=y)



'''
classifyLogisticRegression
==========================
Builds a Logistic Regression classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyLogisticRegression(datasets, test_train_sets, y_colname, id_colname):

	log_cv = linear_model.LogisticRegressionCV(cv=10, penalty='l1', scoring='recall', solver='liblinear', n_jobs=-1)
	log_cv.fit(test_train_sets['X_train'], test_train_sets['y_train'])
	pred_classes = log_cv.predict(test_train_sets['X_test'])
	pred_probs_train = log_cv.predict_proba(test_train_sets['X_train'])
	pred_probs_test = log_cv.predict_proba(test_train_sets['X_test'])

	print("LOGISTIC REGRESSION")
	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))

	return [pred_classes, pred_probs_train, pred_probs_test]



'''
classifySVM
==========================
Builds an SVM classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifySVM(datasets, test_train_sets, y_colname, id_colname):

	# cross validate to find best hyperparameters
	# raw_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced')
	# search_params = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.01, 1, 10, 100]}
	# svm_cv = grid_search.GridSearchCV(raw_svm, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)
	# svm_cv.fit(test_train_sets['X_train'], test_train_sets['y_train'])
	# print(svm_cv.best_params_)

	# use best hyperparameters in the future (uncomment above and comment out this line to re-cross validate)
	best_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced', kernel='rbf', C=100, gamma=0.0001, probability=True)
	best_svm.fit(test_train_sets['X_train'], test_train_sets['y_train'])

	pred_classes = best_svm.predict(test_train_sets['X_test'])
	pred_probs_train = best_svm.predict_proba(test_train_sets['X_train'])
	pred_probs_test = best_svm.predict_proba(test_train_sets['X_test'])

	print("SVM")
	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))

	return [pred_classes, pred_probs_train, pred_probs_test]



'''
classifyNearestNeighbors
==========================
Builds a nearest neighbors classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyNearestNeighbors(datasets, test_train_sets, y_colname, id_colname):

	knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
	knn.fit(test_train_sets['X_train'], test_train_sets['y_train'])
	pred_classes = knn.predict(test_train_sets['X_test'])
	pred_probs_test = knn.predict_proba(test_train_sets['X_test'])
	pred_probs_train = knn.predict_proba(test_train_sets['X_train'])

	print("K-NEAREST NEIGHBORS")
	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))

	return [pred_classes, pred_probs_train, pred_probs_test]



'''
classifyNaiveBayes
==========================
Builds a naive Bayes classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyNaiveBayes(datasets, test_train_sets, y_colname, id_colname):

	nb_model = naive_bayes.MultinomialNB(alpha=0.5)
	nb_model.fit(test_train_sets['X_train'], test_train_sets['y_train'])
	pred_classes = nb_model.predict(test_train_sets['X_test'])
	pred_probs_test = nb_model.predict_proba(test_train_sets['X_test'])
	pred_probs_train = nb_model.predict_proba(test_train_sets['X_train'])

	print("NAIVE BAYES")
	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))

	return [pred_classes, pred_probs_train, pred_probs_test]



'''
classifyRandomForest
==========================
Builds a random forest classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyRandomForest(datasets, info_df):
	rand_forest = ensemble.RandomForestClassifier(n_estimators=20, oob_score=True, n_jobs=-1, verbose=5, class_weight='balanced')
	search_params = {'max_depth':[10,100,1000]}
	rand_forest_cv = grid_search.GridSearchCV(rand_forest, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)

	rand_forest_cv.fit(datasets['X_train'], datasets['y_train'])
	preds = rand_forest_cv.predict(datasets['X_test'])
	evaluateModel(rand_forest_cv, datasets, preds)
	# printNonZeroFeatures(rand_forest_cv, datasets)



'''
printNonZeroCoefficients
========================
Print out the coefficients from regularized model that haven't been set to zero,
to help with evaluating models.
'''
def printNonZeroCoefficients(model, datasets):
	coefs = pd.DataFrame({'feature':datasets['feature_names'], 'coefficient':model.coef_[0]})
	print(coefs[coefs['coefficient'] > 0])



'''
printNonZeroFeatures
========================
Print out the features from a random forest model that haven't been set to zero,
to help with evaluating models.
'''
def printNonZeroFeatures(model, datasets):
	coefs = pd.DataFrame({'feature':datasets['feature_names'], 'coefficient':model.feature_importances_})
	print(coefs[coefs['coefficient'] > 0])



'''
matchPredsToInfoDF
==================
Uses the match id to match predictions up with the original agenda items
to get a better sense of what the model is catching/missing.
'''
def matchPredsToInfoDF(true_y, pred_y, ids, info_df):
	# build df of preds, with id as index
	preds_df = pd.DataFrame({'true_y': true_y, 'pred_y': pred_y}, index=ids)
	combined_df = info_df.join(preds_df, how='inner')

	print("============MATCHED SUCCESSFULLY===========")
	print(combined_df.loc[(combined_df['true_y'] == 1) & (combined_df['pred_y']==1), 'item_text'])

	print("============FALSE NEGATIVE===========")
	print(combined_df.loc[(combined_df['true_y'] == 1) & (combined_df['pred_y']==0), 'item_text'])

	print("============FALSE POSITIVE===========")
	print(combined_df.loc[(combined_df['true_y'] == 0) & (combined_df['pred_y']==1), 'item_text'])



'''
compareClassifiers
==================
Given predictions from several different classifiers,
see how they compare.
'''
def compareClassifiers(datasets, test_train_sets, preds_dict):
	preds_df = pd.DataFrame({'true_y': test_train_sets['y_test']}, index=test_train_sets['ids_test'])

	for key in preds_dict:
		preds_df[key] = preds_dict[key]

	preds_df = datasets['info_df'].join(preds_df, how='inner')

	# how many do all predict correctly as true
	print("All True Positive")
	print(preds_df[preds_df.eq(1, axis='index').all(1)].shape)

	print("SVM & NB true positive")
	print(preds_df[(preds_df['true_y'] == 1) & (preds_df['svm'] == 1) & (preds_df['nb'] == 1)].shape)

	print("NB & SVM false negative")
	print(preds_df[(preds_df['true_y'] == 1) & (preds_df['nb'] == 0) & (preds_df['svm'] == 0)].shape)


	print("SVM true positive, nb false negative")
	print(preds_df[(preds_df['true_y'] == 1) & (preds_df['svm'] == 1) & (preds_df['nb'] == 0)].shape)

	print("naive bayes true positive, SVM false negative")
	print(preds_df[(preds_df['true_y'] == 1) & (preds_df['nb'] == 1) & (preds_df['svm'] == 0)].shape)

	print("KNN true positive, NB & SVM false negative")
	print(preds_df[(preds_df['true_y'] == 1) & (preds_df['knn'] == 1) & (preds_df['nb'] == 0) & (preds_df['svm'] == 0)].shape)

	print("-----------------")

	print("SVM true negative, NB false positive")
	print(preds_df[(preds_df['true_y'] == 0) & (preds_df['nb'] == 1) & (preds_df['svm'] == 0)].shape)

	print("NB true negative, SVM false positive")
	print(preds_df[(preds_df['true_y'] == 0) & (preds_df['nb'] == 0) & (preds_df['svm'] == 1)].shape)

	# print("========= NB & SVM FALSE NEGATIVES==========")
	# print(preds_df.loc[(preds_df['true_y'] == 1) & (preds_df['nb'] == 0) & (preds_df['svm'] == 0), 'item_text'])




'''
combineClassifiers
==================
Given a dict of predicted probabilities that an observation is positive from several different classifiers,
try building an ensemble classifier that aggregrates together these estimates.
'''
def combineClassifiers(datasets, test_train_sets, pred_probs_train, pred_probs_test):
	X_train = np.column_stack((pred_probs_train['lr'], pred_probs_train['svm'], pred_probs_train['nb'], pred_probs_train['knn']))
	X_test = np.column_stack((pred_probs_test['lr'], pred_probs_test['svm'], pred_probs_test['nb'], pred_probs_test['knn']))

	# X_train = np.column_stack((pred_probs_train['lr'], pred_probs_train['svm'], pred_probs_train['nb']))
	# X_test = np.column_stack((pred_probs_test['lr'], pred_probs_test['svm'], pred_probs_test['nb']))

	# try a simple logistic regression
	print("COMBINED LOG REGRESSION CV")
	log_cv = linear_model.LogisticRegressionCV(cv=10, penalty='l1', scoring='recall', solver='liblinear', n_jobs=-1)
	log_cv.fit(X_train, test_train_sets['y_train'])
	pred_classes = log_cv.predict(X_test)

	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))
	print(log_cv.coef_)

	# try naive bayes
	print("COMBINED NAIVE BAYES")
	nb_model = naive_bayes.GaussianNB()
	nb_model.fit(X_train, test_train_sets['y_train'])
	pred_classes = nb_model.predict(X_test)

	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))

	# try an SVM
	print("COMBINED SVM")
	# raw_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced')
	# search_params = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.01, 1, 10, 100]}
	# svm_cv = grid_search.GridSearchCV(raw_svm, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)
	# svm_cv.fit(X_train, test_train_sets['y_train'])
	# print(svm_cv.best_params_)

	best_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced', kernel='rbf', C=1, gamma=1)
	best_svm.fit(X_train, test_train_sets['y_train'])
	pred_classes = best_svm.predict(X_test)

	print(metrics.classification_report(test_train_sets['y_test'], pred_classes))
	print(metrics.confusion_matrix(test_train_sets['y_test'], pred_classes))


if __name__ == '__main__':
    main()
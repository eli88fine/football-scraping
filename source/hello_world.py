import os
import sys

import requests
import openpyxl
import sklearn.svm
import sklearn.model_selection
import numpy as np

outputDirPrefix = os.path.join(os.sep, 'mnt')

dataDirPrefix = os.path.join(os.sep, 'football_scraping', 'data')

def load_machine_input_file():
    wb = openpyxl.load_workbook(filename=os.path.join(dataDirPrefix, 'machine_input.xlsx'), data_only=True)

    current_sheet = wb['Sheet']
    training_data = list()
    ground_truths = list()
    for this_row_index, this_row in enumerate(tuple(current_sheet.rows)):
        this_row_values = [cell.value for cell in this_row]
        if this_row_index == 0:
            feature_names = this_row_values[1:]
        
        if this_row_index >= 3: # actual data
            ground_truths.append(this_row_values[1])
            training_data.append(this_row_values[2:])
        #for this_column_index, this_cell in enumerate(this_row):
            
                
    

    return feature_names, training_data, ground_truths

def shuffle_data(training_data, ground_truths):
    """
    For use during nested cross-validation, will shuffle the order of the data. Returns copies of the data that have been shuffled
    """
    shuffled_training = np.asarray(training_data)
    shuffled_truths = np.asarray(ground_truths)
    n_samples = len(ground_truths)
    order = np.random.permutation(n_samples)
        
    return shuffled_training[order], shuffled_truths[order]
    
def custom_scoring_function(estimator, training_data, ground_truths):
    betting_confidence_threshold = 0.975
    minimum_fraction_above_betting_threshold = 0.50
    probabilities = estimator.predict_proba(training_data[:])
    
    n_samples = np.size(ground_truths)
    
    highest_confidences = np.amax(probabilities, axis=1)
    #pqrint (highest_confidences)
    indices_passing_confidence_threshold = np.where(highest_confidences >= betting_confidence_threshold)[0]
    #pqrint (np.size(indices_passing_confidence_threshold))
    number_of_samples_passing_confidence_threshold = np.size(indices_passing_confidence_threshold)
    fraction_of_samples_passing_confidence_threshold = number_of_samples_passing_confidence_threshold / n_samples
    #pqrint (probabilities)
    if fraction_of_samples_passing_confidence_threshold >= minimum_fraction_above_betting_threshold:
        number_of_correct_predictions = 0
        for this_index, this_confidence in enumerate(highest_confidences):
            if this_confidence >= betting_confidence_threshold:
                #pqrint ("Ground Truth: %s, probabilities: %s" % (ground_truths[this_index], probabilities[this_index]))
                if probabilities[this_index][ground_truths[this_index]] > betting_confidence_threshold:
                    number_of_correct_predictions += 1
        fraction_of_correct_high_confidence_predictions = number_of_correct_predictions / number_of_samples_passing_confidence_threshold
        #pqrint ("%s fraction passed confidence threshold, and %s fraction of those were correct." % (fraction_of_samples_passing_confidence_threshold, fraction_of_correct_high_confidence_predictions))

        return (fraction_of_correct_high_confidence_predictions * 5 + fraction_of_samples_passing_confidence_threshold) / 6
    else:
        #pqrint ("Only %s fraction passed confidence threshold, returning score of 0" % fraction_of_samples_passing_confidence_threshold)
        return 0
    
    
def run_nested_cross_validation(training_data, ground_truths, C=1.0, num_nests=20, num_folds=5):
    estimator = sklearn.svm.SVC(kernel='linear', probability=True)
    nest_scores = list()
    for i in range(num_nests):
        this_search = sklearn.model_selection.GridSearchCV(estimator, {'C': [C]}, cv=num_folds, scoring=custom_scoring_function)
        shuffled_training, shuffled_truths = shuffle_data(training_data, ground_truths)
        this_search.fit(shuffled_training, shuffled_truths)
        
        nest_scores.append(this_search.best_score_)
    return np.mean(nest_scores)
    
def run_grid_search(training_data, ground_truths):
    for this_C in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        this_score = run_nested_cross_validation(training_data, ground_truths, C=this_C)
        print ("C: %s, score: %s" % (this_C, this_score))

    
def train_classifier(training_data, ground_truths):
    classifier = sklearn.svm.SVC(C=1.0, kernel='linear', probability=True)
    classifier.fit(training_data, ground_truths)
    return classifier
    
# url = 'https://assets.digitalocean.com/articles/eng_python/beautiful-soup/mockturtle.html'
if __name__ == "__main__":
    feature_names, training_data, ground_truths = load_machine_input_file()
    classifier = train_classifier(training_data, ground_truths)
    print (classifier.coef_)
    print (classifier.predict_proba([[-10]]))
    
    #run_nested_cross_validation(training_data, ground_truths)
    run_grid_search(training_data, ground_truths)
    
    
    #print (training_data)
# page = requests.get(url)

# pqrint ("Status code: %s" % page.status_code)

#pqrint (page.text)

#wb = openpyxl.Workbook()


# wb.save(filename=os.path.join(outputDirPrefix, 'test.xlsx'))
# with open(os.path.join(outputDirPrefix, 'testfile3.txt'), 'w') as f:
    # pass
run.py is used to get the results for the test dataset
for proper functioning we will need the complete path of the test dataset along with the pickle files for xtrain and ytrain to train the classifier
the script saves a file named "output.csv" which contains the filenames and the predicted class label
if somehow the script doesn't run, try removing the @call_parse line before the main() definition
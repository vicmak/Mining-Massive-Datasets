# Databricks notebook source
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel, LogisticRegressionWithLBFGS
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
import random
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

file1 = "/FileStore/tables/mql7k6ul1484733936144/subject101.dat"
file2 = "/FileStore/tables/mql7k6ul1484733936144/subject102.dat"
file3 = "/FileStore/tables/mql7k6ul1484733936144/subject103.dat"
file4 = "/FileStore/tables/mql7k6ul1484733936144/subject104.dat"
file5 = "/FileStore/tables/mql7k6ul1484733936144/subject105.dat"
file6 = "/FileStore/tables/mql7k6ul1484733936144/subject106.dat"
file7 = "/FileStore/tables/mql7k6ul1484733936144/subject107.dat"
file8 = "/FileStore/tables/mql7k6ul1484733936144/subject108.dat"

#filenames = [file1, file2, file3]
filenames = [file1, file2, file3, file4, file5, file6, file7, file8]

def slice_data(line):
    raw_list = line.split(" ")
    indices = {0,7,8,9,16,17,18,19,24,25,26,33,34,35,36,41,42,43,50,51,52,53}
    sliced_list = [i for j, i in enumerate(raw_list) if j not in indices]
    return sliced_list 

lines = sc.emptyRDD()
  
for i in range(0, len(filenames)):
    file_content = sc.textFile(filenames[i])
    lines = lines.union(file_content)

lines = lines.map(slice_data)

features_number = len(lines.first()) - 1 #the first is the labeled class

summed_data = [[sc.accumulator(0), sc.accumulator(0)] for y in range(features_number)]   

Nans = sc.accumulator(0)

def sum_data(sliced_list):
    print ("here")
    for i in range(1, len(sliced_list)):
        print i
        if sliced_list[i] != "NaN":
            summed_data[i-1][0] += float(sliced_list[i])
            summed_data[i-1][1] += 1
        else:
            Nans.add(1)
            
lines.foreach(sum_data)

features_means = [0 for i in range (features_number)]

for i in range(0, features_number):
    features_means[i] = summed_data[i][0].value / summed_data[i][1].value
    print features_means[i]

labels = ["1","2","3","4","5","6","7","9","10","11","12","13","16","17","18","19","20","24"]
encoded_labels = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10:8, 11:9, 12:10, 13:11, 16:12, 17:13, 18:14, 19:15, 20:16, 24:17}

def fill_missing_values(sliced_list):
    features = [0 for i in range (features_number)]
    label = int(sliced_list[0])
    for i in range(0, features_number):
        if sliced_list[i+1] != "NaN":
            features[i] = float(sliced_list[i+1])
        else:
            features[i] = features_means[i]
        #if  features[i] < 0: #To use with Naive Bayes
         #   features[i] = features[i] * (-1)
    
    lp = LabeledPoint(encoded_labels[label], features) 
    return lp
   

def filter_labels(line):
    return line[0] in labels
        
#filled_data = lines.filter(filter_labels).map(fill_missing_values)


def run_leave_one_out():
    count = len(filenames)-1
    error = 0.0
    for i in range (0, count):
        data = sc.emptyRDD()
        for j in range (0, count):
            if j != i:
                current_file = sc.textFile( filenames[j]) #original file
                current_lines = current_file.map(slice_data)
                current_filled = current_lines.filter(filter_labels).map(fill_missing_values)
                data = data.union(current_filled)
        current_test = sc.textFile(filenames[i]).map(slice_data).filter(filter_labels).map(fill_missing_values)
        error = error + get_error(data, current_test)
    return error/float(count)  

  
def get_error(training, test):

    model = LogisticRegressionWithLBFGS.train(training,numClasses=18)

    # Evaluating the model on training data
    labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
    ERR = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
    print("Training Error = " + str(ERR))
    return ERR
   
  
  
print ("Running LOO")
print run_leave_one_out()

print("end")
  
    #print "Naive Bayes"
    # Train a naive Bayes model.
    #model = NaiveBayes.train(training, 1.0)
    # Make prediction and test accuracy.
    #predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    #err = 1.0 * predictionAndLabel.filter(lambda (x, v): x != v).count() / test.count()
    #return err

#model = RandomForest.trainClassifier(trainingData, numClasses=18, categoricalFeaturesInfo={},
#                                     numTrees=3, featureSubsetStrategy="auto",
#                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
#predictions = model.predict(testData.map(lambda x: x.features))
#labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
#testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
#print('Test Error = ' + str(testErr))


# COMMAND ----------



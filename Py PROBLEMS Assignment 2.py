# Databricks notebook source
# MAGIC %md
# MAGIC # Assignment Problem 2: Analyze Forest Coverage 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Instructions 
# MAGIC **This is an Open-ended problem. I'm not looking for one correct answer, but an organized analysis report on the data. **
# MAGIC 
# MAGIC This is a very clean dataset great for classification. The data file contains 581,012 lines, each containing 55 fields. The first 54 fields are properties of a certain place on earth, the 55th field is the type of land coverage. Details of the fields in the README file below. 
# MAGIC 
# MAGIC 1. Use Spark to parse the file, prepare data for classification;
# MAGIC 1. Show some basic statistics of the data fields
# MAGIC 1. Build a Random Forest model in Spark to analyze the data. 
# MAGIC 1. Split the dataset to 70% and 30% for training and test dataset.  
# MAGIC 1. Train differnt classificiers and observe the performance/error rate of them. 
# MAGIC 1. Use Spark to do your calculations, then use dataframes to draw some plots. Describe each plot briefly and draw some conclusions.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### How to work on and hand in this assignment
# MAGIC Write your analysis report, and send me the PDF of your Notebook before the assignment is due. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the data into an RDD

# COMMAND ----------

#read txt file, gzipped files will be auto-unzipped
myDataRDD = sc.textFile("/mnt/mlonspark/covtype.data.gz")
myReadmeRDD = sc.textFile("/mnt/mlonspark/covtype.info")

print myDataRDD.count()
myDataRDD.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Readme File

# COMMAND ----------

for l in myReadmeRDD.collect():
  print l
  

# COMMAND ----------

clean_data = myDataRDD.map(lambda x: [int(i) for i in x.split(',')])

clean_data.count()

# COMMAND ----------

wilderness_Area = ["Wilderness_Area_" + str(i) for i in range(1,5)]
soil_Type = ["Soil_Type_" + str(i) for i in range(1,41)]

features = [ "Elevation", "Aspect" , "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]+ wilderness_Area + soil_Type +["label"]

dataDF = sqlContext.createDataFrame(clean_data, features)

display(dataDF)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler =  VectorAssembler(inputCols = features[:-1], outputCol = "features")

data = assembler.transform(dataDF).select("label", "features")
data.show(3)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

training, testing = data.randomSplit([0.7, 0.3])
# Train a RandomForest model.
forest = RandomForestClassifier(maxBins=32, numTrees=20, maxDepth=12, labelCol="label", featuresCol="features")

#maxBins=50,  numTrees=4,  maxDepth=3  => 66.2
#maxBins=100, numTrees=8,  maxDepth=6  => 68
#        50            10           10 => 73
#        50            20           10 => 75
#        50            20           8  => 72
#        50            20           12 => 76
#        32            20           12 => 77

pipeline_forest = Pipeline().setStages([forest])

model_forest = pipeline_forest.fit(training)

predictions = model_forest.transform(testing)

predictions.take(10)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print(evaluator.evaluate(predictions))

#grid search

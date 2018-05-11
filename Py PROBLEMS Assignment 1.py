# Databricks notebook source
# MAGIC %md
# MAGIC # Assignment Problem 1: Analyze Power Usage Data 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Instructions 
# MAGIC **This is an Open-ended problem. I'm not looking for one correct answer, but an organized analysis report on the data. **
# MAGIC 
# MAGIC We will use a dataset from one smartmeter to analyze the energy consumption pattern of a house. Analyze the electricity usage pattern and see what conclusion you can draw about the resident?
# MAGIC 
# MAGIC Note:
# MAGIC 
# MAGIC 1. You need to pay attention to missing data;
# MAGIC 2. calculate some aggregate values of energy usage and observe different type of trends (e.g. pattern in a day, pattern in a week, pattern in a year, etc);
# MAGIC 3. Use Spark to do your calculations, then use dataframes to draw some plots. Describe each plot briefly and draw some conclusions;
# MAGIC 4. You only need to use the simple Spark transformations and actions covered in class, no need to use machine learning methods yet. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### How to work on and hand in this assignment
# MAGIC Simply clone this Notebook to your own directory. Write your analysis report, and send me the link (see address bar) of your Notebook before the assignment is due. 
# MAGIC 
# MAGIC If you prefer to do it in Python (or R), you can create a Python/R Notebook and send me the link to it. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description of the Dataset
# MAGIC Source: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#
# MAGIC 
# MAGIC This archive contains 2075259 measurements gathered between December 2006 and November 2010 (47 months). 
# MAGIC 
# MAGIC Notes: 
# MAGIC 
# MAGIC 1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3. 
# MAGIC 
# MAGIC 2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.
# MAGIC 
# MAGIC 
# MAGIC Attribute Information:
# MAGIC 
# MAGIC 1.date: Date in format dd/mm/yyyy 
# MAGIC 
# MAGIC 2.time: time in format hh:mm:ss 
# MAGIC 
# MAGIC 3.global_active_power: household global minute-averaged active power (in kilowatt) 
# MAGIC 
# MAGIC 4.global_reactive_power: household global minute-averaged reactive power (in kilowatt) 
# MAGIC 
# MAGIC 5.voltage: minute-averaged voltage (in volt) 
# MAGIC 
# MAGIC 6.global_intensity: household global minute-averaged current intensity (in ampere) 
# MAGIC 
# MAGIC 7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered). 
# MAGIC 
# MAGIC 8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light. 
# MAGIC 
# MAGIC 9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the data into an RDD

# COMMAND ----------

#read txt file, gzipped files will be auto-unzipped
myRDD = sc.textFile("/mnt/mlonspark/household_power_consumption.txt.gz")

print myRDD.count()

myRDD.take(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Start Your Analysis Here

# COMMAND ----------

#filter data
tuples = myRDD.map(lambda x: x.split(';')).filter(lambda x: not '' in x)



# COMMAND ----------

# helper functions
from datetime import datetime

def getDate(date):
  return datetime.strptime(date, '%d/%m/%Y')

tuples = data.map(lambda x: (getDate(x[0]), float(x[2])))

power_used_per_day = tuples.reduceByKey(lambda x,y: x+y).sortByKey()
tupleDF = sqlContext.createDataFrame(power_used_per_day,('Date','Total Power Per Day'))
display(tupleDF)

  

# COMMAND ----------

total_power_by_day_of_week = power_used_per_day.map(lambda x: (x[0].weekday(), (x[1], 1.0)))

total_power_by_day_of_week = total_power_by_day_of_week.reduceByKey(lambda x,y: (x[0]+y[0], x[1]+ y[1])).sortByKey()

total_power_by_day_of_week = total_power_by_day_of_week.map(lambda x: (x[0], x[1][0]/x[1][1]))

total_power_by_day_of_week.take(10)

tupleDF = sqlContext.createDataFrame(total_power_by_day_of_week,('Date','Total Power Per Day of the week'))
display(tupleDF)

# COMMAND ----------

days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

total_power_by_day_of_week = total_power_by_day_of_week.map(lambda x: (days[x[0]], x[1]))

total_power_by_day_of_week.collect()
display(total_power_by_day_of_week.toDF())

#Importing Requirements
from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql import functions as F

from pyspark.sql.types import IntegerType
from functools import reduce

sc = SparkContext()
spark = SparkSession(sc)

data_dir = "gs://bdl2021_final_project/nyc_tickets_train.csv"
# data_dir = "/content/drive/MyDrive/MyPC/Sem 8/BD LAB PROJECT"
# data_dir = "/content/drive/MyDrive/BDLProjectDataSet"

df = spark.read.format("csv").option("recursiveFileLookup", "true").option("pathGlobFilter", "*.csv").load(data_dir,inferSchema=True, header = True)

drop_cols = ['No Standing or Stopping Violation',
             'Hydrant Violation', 'Double Parking Violation', 'Latitude',
             'Longitude', 'Community Board', 'Community Council', 'Census Tract',
             'BIN', 'BBL', 'NTA','Plate ID','Summons Number','Vehicle Expiration Date','Time First Observed','Date First Observed','From Hours In Effect','To Hours In Effect','Unregistered Vehicle?','Days Parking In Effect','House Number','Violation Legal Code','Issuer Code']

req_cols = ["Registration State", "Plate Type", "Issue Date",  "Violation Code", "Vehicle Body Type",  "Vehicle Make",  "Issuing Agency", "Street Code1",  "Street Code2", "Street Code3", 
            "Issuer Command",  "Issuer Squad",  "Violation Time",  "Violation In Front Of Or Opposite", "Street Name",  "Intersecting Street",  "Law Section", "Sub Division", "Vehicle Color", 
            "Vehicle Year", "Meter Number", "Feet From Curb", "Violation Post Code", "Violation Description"]

drop_cols = list(map(lambda x: "_".join(x.split()), drop_cols))
req_cols = list(map(lambda x: "_".join(x.split()), req_cols))

oldColumns = df.schema.names
df = reduce(lambda df, idx: df.withColumnRenamed(oldColumns[idx], "_".join(oldColumns[idx].split())), range(len(oldColumns)), df)

df.drop(*drop_cols)

string_cols = req_cols.copy()
string_cols.remove("Violation_Code")
string_cols.remove("Street_Code1")
string_cols.remove("Street_Code2")
string_cols.remove("Street_Code3")
string_cols.remove("Vehicle_Year")
string_cols.remove("Feet_From_Curb")


#########################################################################
#Remove Extra two columns
string_cols.remove("Issue_Date")
string_cols.remove("Violation_Time")

#Defining two udfs
@F.udf(returnType=IntegerType())
def dateParse(dstr):
  BrStr = dstr.split("/")
  NewBrStr = [BrStr[2], BrStr[0], BrStr[1]]
  DateAsNo = int("".join(NewBrStr))
  return DateAsNo

@F.udf(returnType=IntegerType())
def timeParse(tstr):
  try:
    NoPart = tstr[:-1]
    dayLight = tstr[-1]
    if(dayLight == 'A'):
      return(int(NoPart))
    else:
      MM = NoPart[-2:]
      HH = str(int(NoPart[:-2])%12+12)
      TimeAsNo = int("".join([HH,MM]))
      return TimeAsNo
  except Exception:
    return None
    
#Pass into both udf's
df = df.withColumn("Issue_Date", dateParse("Issue_Date"))
df = df.withColumn("Violation_Time", timeParse("Violation_Time"))
#########################################################################


#Assign the Class Label column as label
indexer = StringIndexer(inputCol="Violation_County", outputCol="Violation_County_ind")
df = indexer.fit(df).transform(df)

df = df.withColumn("label", df["Violation_County_ind"])
df.select(F.col("Violation_Code").cast('int').alias("Violation_Code"))
df.select(F.col("Street_Code1").cast('int').alias("Street_Code1"))
df.select(F.col("Street_Code2").cast('int').alias("Street_Code2"))
df.select(F.col("Street_Code3").cast('int').alias("Street_Code3"))
df.select(F.col("Vehicle_Year").cast('int').alias("Vehicle_Year"))
df.select(F.col("Feet_From_Curb").cast('float').alias("Feet_From_Curb"))


# Label Encoding of Classes and removing the original column
indexers = [StringIndexer(inputCol=column, outputCol=column+"_ind").setHandleInvalid(value="skip") for column in string_cols]
# indexer = StringIndexer(inputCol=string_cols, outputCol=string_cols)
req_cols = [col+"_ind" for col in string_cols] + [col for col in req_cols if col not in string_cols]

imputer = Imputer()
imputer.setStrategy("mode")
imputer.setInputCols(req_cols)
imputer.setOutputCols([col+"_imp" for col in req_cols])
req_cols = [col+"_imp" for col in req_cols]

assembler1 = VectorAssembler(inputCols = req_cols, outputCol = "features_old")
scaler = StandardScaler(inputCol="features_old", outputCol="scaledFeatures",withStd=True, withMean=True)
assembler2 = VectorAssembler(inputCols = ["scaledFeatures"], outputCol = "features")

# Construct a new Logistic Regression object and fit the training data.
rf = RandomForestClassifier()
#pipeline
pipe = Pipeline(stages = indexers + [imputer,assembler1,scaler,assembler2, rf])
# model = pipe.fit(df)

# Create a grid of multiple values of the hyper-parameter regParam
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth,[20,30]).addGrid(rf.numTrees,[50,100]).build()

#Create a CrossValidator Object
obj = TrainValidationSplit(estimator=pipe, 
                           estimatorParamMaps=paramGrid, 
                           evaluator=MulticlassClassificationEvaluator(metricName = 'accuracy'),
                           trainRatio=0.9,
                           seed = 2021)


#Train the model with the CrossValidator Object 
trained_model = obj.fit(df)

#SAVING...
trained_model.bestModel.save("gs://bigdata_lab7/bestModel_RandomForest") ##Change Directory

# Acquire and print the best model details from the CrossValidator object
print("\033[1mValidation Results\033[0m")
val = list(zip(trained_model.validationMetrics, trained_model.getEstimatorParamMaps()))
# print("Accuracy: " + str(max(list(trained_model.validationMetrics)))) 
print(val)
# print('Best Max depth: ',trained_model.bestModel.getMaxDepth())
# print('Best Numer of Trees: ',trained_model.bestModel.getNumTrees())

# #Testing on full data
# prediction = trained_model.transform(df)
  
# #Create an evaluation object for the model using accuracy and f1 score
# evaluator_acc = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol="label", metricName="accuracy")

# evaluator_f1 = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol="label", metricName="f1")

# # Print the Evaluation Result
# print("\033[1mBest Model Results on Full Training Data\033[0m")
# print("Accuracy on Full training data:", evaluator_acc.evaluate(prediction))
# print("Average F1 score on Full training data:", evaluator_f1.evaluate(prediction))
#Importing Requirements
from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, StringIndexerModel, Imputer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from pyspark.sql.types import IntegerType
from functools import reduce


spark = SparkSession.builder.appName("Poject_NYC_tickets").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

#address of Kafka server
IP_ = "10.128.0.48:9092"

#read from kafka
df = spark.readStream \
     .format("kafka") \
     .option("kafka.bootstrap.servers",IP_) \
     .option("subscribe","PROJECT") \
     .load()


#All features
features = ['Summons Number','Plate ID','Registration State','Plate Type','Issue Date','Violation Code','Vehicle Body Type','Vehicle Make',
            'Issuing Agency','Street Code1','Street Code2','Street Code3','Vehicle Expiration Date','Issuer Code','Issuer Command','Issuer Squad',
            'Violation Time','Time First Observed','Violation_County','Violation In Front Of Or Opposite','House Number','Street Name','Intersecting Street',
            'Date First Observed','Law Section','Sub Division','Violation Legal Code','Days Parking In Effect','From Hours In Effect','To Hours In Effect',
            'Vehicle Color','Unregistered Vehicle?','Vehicle Year','Meter Number','Feet From Curb','Violation Post Code','Violation Description',
            'No Standing or Stopping Violation','Hydrant Violation','Double Parking Violation','Latitude','Longitude','Community Board','Community Council',
            'Census Tract','BIN','BBL','NTA']

features = list(map(lambda x: "_".join(x.split()),features))

#add feature columns to DataFrame
columns = F.split(df.value,',')
for i in range(len(features)):
    df = df.withColumn(features[i],columns[i].cast('string'))


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

#cast columns to appropriate dtype
df =  df.withColumn("Violation_Code", F.col("Violation_Code").cast('int'))
df =  df.withColumn("Street_Code1", F.col("Street_Code1").cast('int'))
df =  df.withColumn("Street_Code2", F.col("Street_Code2").cast('int'))
df =  df.withColumn("Street_Code3", F.col("Street_Code3").cast('int'))
df =  df.withColumn("Vehicle_Year", F.col("Vehicle_Year").cast('int'))
df =  df.withColumn("Feet_From_Curb", F.col("Feet_From_Curb").cast('float'))


#load label indexer and transform df
indexer = StringIndexerModel.load('gs://avinashbagali/Label_Indexer/')
df = indexer.transform(df)
df = df.withColumn("label", df["Violation_County_ind"])


#load best model and transform df
best_model = PipelineModel.load("gs://avinashbagali/bestModel_RandomForest/")
df = best_model.transform(df)

#result df -> ["Violation_County","label","prediction"]
result_df = df[["Violation_County","label","prediction"]]


# function to use with foreachBatch to compute accuracy and F1-score
def Metrics(df, epoch_id):
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol = "prediction",
                                                      labelCol="label",
                                                      metricName="accuracy")

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol = "prediction",
                                                     labelCol="label",
                                                     metricName="f1")

    acc = evaluator_acc.evaluate(df)
    f1  = evaluator_f1.evaluate(df)


    print("+------------------------------+")
    print("|Number of rows in batch {} = {}|".format(epoch_id,df.count()))
    print("|  Accuracy on batch {} is {} |".format(epoch_id,round(acc,3)))
    print("|  F1 score on batch {} is {} |".format(epoch_id,round( f1,3)))
    print("+------------------------------+")


#print result_df -> ["Violation_County","label","prediction"]
query1 = result_df \
       .writeStream \
       .outputMode("append") \
       .format("console") \
       .start()


#print Accuracy and F1-score 
query2 = result_df \
       .writeStream \
       .format("console") \
       .foreachBatch(Metrics) \
       .start()


query1.awaitTermination()
query2.awaitTermination()


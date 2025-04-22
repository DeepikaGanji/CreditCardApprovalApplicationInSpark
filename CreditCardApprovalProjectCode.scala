// SparkScala Final Project - Credit Card Approval Application in Spark

// Loading the dataset into respective dataframes and performing the initial checks

val applicationDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header","true").load("SparkScalaFinalProject/application_records.csv")

// printing the schema

applicationDF.printSchema()

// Count of records

applicationDF.count()

// checking few records of the dataframe

applicationDF.show(5, false)

// Descriptions of the columns to identify columns that show count less than total records due to missing values

import org.apache.spark.sql.functions._

applicationDF.describe().show()

// Checking missing values of the columns that have missing values get the count

applicationDF.filter($"OCCUPATION_TYPE".isNull).count()

// Droping the rows that have null values and verifying again

val applicationDF2 = applicationDF.na.drop()

applicationDF2.describe().show()

// Reading the next input file into a dataframe

val creditDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header","true").load("SparkScalaFinalProject/credit_records.csv")

// Checking by printing the schema

creditDF.printSchema()

// Count of records

creditDF.count()

// Show a few records of the second dataframe

creditDF.show(5, false)

// To get descriptions of the coulmns

creditDF.describe().show()

// Check for duplicate records in this dataset based on ID column and MONTHS_BALANCE
 
import org.apache.spark.sql.SparkSession

// Removing duplicates based on two columns:ID column and MONTHS_BALANCE 

val creditDF2 = creditDF.dropDuplicates("CR_ID", "MONTHS_BALANCE")

creditDF2.count() // no duplicates found

// Joining the two dataframes on ID column

val creditDF3 = creditDF2.join(applicationDF2,creditDF2("CR_ID")===applicationDF2("APP_ID"), "inner")

// Get the count of records in the joined dataframe 

creditDF3.count()

// See the schema of the joined dataframe

creditDF3.printSchema()

// Adding a column 'label' and set its value to 0.0 or 1.0 based on the value of the column STATUS If value of STATUS is ‘C’, ‘X’, ‘0’, ‘1’ then set label value to 0.0 and for all other values of STATUS set label value to 1.0

val creditDF4 = creditDF3.withColumn("label", when(col("STATUS").isin("C", "X", "0", "1"), lit(0.0)).otherwise(lit(1.0)))

creditDF4.show(5)

// considering any record/application as good if the days due is 0 to 60 days and any applicaiton/record with more days due than 60 can be labeled as bad 
// Performing data transformations

// Bucketizer for CNT_CHILDREN and CNT_FAM_MEMBERS
val childrenSplits = Array(0.0, 1.0, 2.0, 4.0)
val familySplits = Array(1.0, 2.0, 3.0, 4.0, 5.0)

import org.apache.spark.ml.feature.Bucketizer

val bucketizer = new Bucketizer().
  setInputCols(Array("CNT_CHILDREN", "CNT_FAM_MEMBERS")).
  setOutputCols(Array("childnum", "familynum")).
  setSplitsArray(Array(childrenSplits, familySplits))

import org.apache.spark.ml.feature.QuantileDiscretizer

// QuantileDiscretizer for AMT_INCOME_TOTAL, DAYS_BIRTH, and DAYS_EMPLOYED
val incomeDiscretizer = new QuantileDiscretizer().
  setInputCols(Array("AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED")).
  setOutputCols(Array("income_category", "age_group", "emp_group")).
  setNumBuckets(7)

import org.apache.spark.ml.feature.StringIndexer

// StringIndexer for categorical columns
val stringIndexer = new StringIndexer().
  setInputCols(Array("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                      "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE")).
  setOutputCols(Array("gender_index", "own_car_index", "own_realty_index", "income_type_index",
                       "education_type_index", "family_status_index", "housing_type_index", "occupation_type_index"))

import org.apache.spark.ml.feature.VectorAssembler

// VectorAssembler to combine all feature columns
val assembler = new VectorAssembler().
  setInputCols(Array("childnum", "familynum", "income_category", "age_group", "emp_group",
                      "gender_index", "own_car_index", "own_realty_index", "income_type_index",
                      "education_type_index", "family_status_index", "housing_type_index", "occupation_type_index")).
  setOutputCol("features")

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Decision Tree Classifier
val dtc = new DecisionTreeClassifier().
  setLabelCol("label").
  setFeaturesCol("features").
  setMaxBins(32)
  
// Building pipeline
val pipeline = new Pipeline().
  setStages(Array(bucketizer, incomeDiscretizer, stringIndexer, assembler, dtc))

// Splitting data into training and testing datasets (70:30)
val Array(trainingData, testData) = creditDF4.randomSplit(Array(0.7, 0.3))

// Training the model
val model = pipeline.fit(trainingData)

// Making predictions
val predictions = model.transform(testData)

// Evaluating the model
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)

println(s"Accuracy: $accuracy")

println(s"Test Error: ${1.0 - accuracy}")

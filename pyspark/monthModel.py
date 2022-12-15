from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

def removeCol(feature_list):
	feature_list.remove("Radiation")
	feature_list.remove("UNIXTime")
	feature_list.remove("Data")
	feature_list.remove("Time")
	feature_list.remove("TimeSunRise")
	feature_list.remove("TimeSunSet")
	feature_list.remove("_c0")	
	print(feature_list)
	return feature_list


def getMonthModel(df_data, month):
	df_data = df_data.filter(F.col("month") == month)
	feature_list = df_data.columns
	feature_list = removeCol(feature_list)

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	rf = RandomForestRegressor(featuresCol="features",labelCol="Radiation")

	trainDF, testDF = df_data.randomSplit([0.8, 0.2], seed=42)

	pipeline = Pipeline(stages=[vecAssembler, rf])
	model = pipeline.fit(trainDF)
	predDF = model.transform(testDF)
	predAndLabel = predDF.select("prediction", "Radiation")
	predAndLabel.show(10)

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("Radiation")

	print("r2", evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print("RMSE:", evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))


if __name__ == "__main__":
	spark = SparkSession.builder.appName("DecisionTree").getOrCreate()
	df_data = spark.read.load("hdfs:///user/maria_dev/project/BDF.csv", format="csv", sep=",", inferSchema=True, header=True)

	for month in range(9, 13):
		getMonthModel(df_data, month)


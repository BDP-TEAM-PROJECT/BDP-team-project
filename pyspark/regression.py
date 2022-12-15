from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("Regression").getOrCreate()
	df_data = spark.read.load("hdfs:///user/maria_dev/project/BDF.csv", format="csv", sep=",", inferSchema=True, header=True)
	feature_list = df_data.columns
	feature_list.remove("Radiation")
	feature_list.remove("UNIXTime")
	feature_list.remove("Data")
	feature_list.remove("Time")
	feature_list.remove("TimeSunRise")
	feature_list.remove("TimeSunSet")
	feature_list.remove("_c0")
	print(feature_list)

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features")
	lr = LinearRegression(featuresCol="features", labelCol="Radiation").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

	trainDF, testDF = df_data.randomSplit([0.8, 0.2], seed=42)

	print(trainDF.cache().count()) # Cache because accessing training data multiple times
	print(testDF.count())

	pipeline = Pipeline(stages=[vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction","Radiation")
	predAndLabel.show(10)

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("Radiation")

	print("r2:", evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print("RMSE:", evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))

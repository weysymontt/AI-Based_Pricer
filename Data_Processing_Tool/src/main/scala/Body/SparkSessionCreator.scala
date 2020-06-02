package Body

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import java.io._
import scala.io.Source


object SparkSessionCreator extends Serializable {
  def createSparkSession(): (SparkSession,Double,Double,Int) = {
    val spark = SparkSession
      .builder
      .appName("pricer")
      .config("spark.master", "local")
      .master("local")
      .getOrCreate()

    val textIterator: Iterator[String] = Source.fromResource("path").getLines()
    val path = textIterator.next()

    // Open input data tables
    val items = spark.read.format("csv").option("header", "true").load(path + "/items.csv")
    val shops = spark.read.format("csv").option("header", "true").load(path + "/shops.csv")
    val sales_train = spark.read.format("csv").option("header", "true").load(path + "/sales_train.csv")

    // Register the DataFrames as global temporary views
    items.createOrReplaceGlobalTempView("items")
    sales_train.createOrReplaceGlobalTempView("sales_train")
    shops.createOrReplaceGlobalTempView("shops")

    val numberOfMonths = (spark.sql(
      f"""SELECT DISTINCT(date_block_num) FROM global_temp.sales_train
         |ORDER BY length(date_block_num) DESC, date_block_num DESC
         |""".stripMargin.replaceAll("\n", " ")).select(col("date_block_num")).first.getString(0).toInt + 1).toDouble

    val numberOfShops = spark.sql(
      f"""SELECT COUNT(shop_id) FROM global_temp.shops
         |""".stripMargin.replaceAll("\n", " ")).select(col("count(shop_id)"))
      .first.toString().replace("[","").replace("]","").toDouble

    val numberOfItems = spark.sql(
      f"""SELECT COUNT(item_id) FROM global_temp.items
         |""".stripMargin.replaceAll("\n", " ")).select(col("count(item_id)"))
      .first.toString().replace("[","").replace("]","").toInt

    (spark,numberOfMonths,numberOfShops,numberOfItems)
  }
}
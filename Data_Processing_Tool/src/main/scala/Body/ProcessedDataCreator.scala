package Body

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import scala.util.control.Breaks._
import java.io.{BufferedWriter, FileWriter}
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer


object ProcessedDataCreator {
  def CreateANNinput(spark:SparkSession,windowExtension:Int,numberOfMonths:Double,numberOfShops:Double,numberOfShopsToUse:Int,numberOfItemsToUse:Int,selectedShops:List[String],selectedItems:List[String]): Unit = {
    println()
    println("Creating the training and testing sets for the Neural Network:")
    val outputFileTrain = new BufferedWriter(new FileWriter(f"../Files/Completely_processed_data/shaped_input_train_$windowExtension%s_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv"))
    val csvWriterTrain = new CSVWriter(outputFileTrain)
    val outputFileTest = new BufferedWriter(new FileWriter(f"../Files/Completely_processed_data/shaped_input_test_$windowExtension%s_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv"))
    val csvWriterTest = new CSVWriter(outputFileTest)
    var csvFields = Array("shop_id", "item_id", "sales_t", "price_t", "relative_shop_size")
    for (i <- 1 until windowExtension) {
      csvFields = csvFields :+ f"sales_t-$i%s"
      csvFields = csvFields :+ f"price_t-$i%s"
      csvFields = csvFields :+ f"salesRelationalCoefficient_t-$i%s"
      csvFields = csvFields :+ f"priceRelationalCoefficient_t-$i%s"
    }
    var structuredDataTrain = new ListBuffer[Array[String]]()
    var structuredDataTest = new ListBuffer[Array[String]]()
    structuredDataTrain += csvFields
    structuredDataTest += csvFields
    var observation = Array[String]()

    val meanShopSize = getMeanShopSize(spark)

    var counter3 = 0
    for (shop <- selectedShops) {
      val relativeShopSize = getRelativeShopSize(spark,shop,selectedShops,meanShopSize)
      var counter = -1
      var counter2 = 0
      for (item <- selectedItems) {
        counter3 += 1
        println("Execution number: " + counter3.toString)
        breakable {
          for (month <- (numberOfMonths.toInt - 1) to 0 by -1) {
            counter += 1
            val productPrice = spark.sql(
              f"""SELECT item_price FROM global_temp.deseasonalizedDf
                 |WHERE shop_id = $shop%s
                 |AND item_id = $item%s
                 |AND date_block_num = $month%s
                 |""".stripMargin.replaceAll("\n", " ")).select(col("item_price"))
              .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble
            val meanPrice = spark.sql(
              f"""SELECT mean_product_price FROM global_temp.product_prices
                 |WHERE item_id = $item%s
                 |AND date_block_num = $month%s
                 |""".stripMargin.replaceAll("\n", " ")).select(col("mean_product_price"))
              .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble
            var productSales = spark.sql(
              f"""SELECT item_cnt_month FROM global_temp.deseasonalizedDf
                 |WHERE shop_id = $shop%s
                 |AND item_id = $item%s
                 |AND date_block_num = $month%s
                 |""".stripMargin.replaceAll("\n", " ")).select(col("item_cnt_month"))
              .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble
            var meanSales = spark.sql(
              f"""SELECT SUM(item_cnt_month) FROM global_temp.deseasonalizedDf
                 |WHERE shop_id <> $shop%s
                 |AND item_id = $item%s
                 |AND date_block_num = $month%s
                 |""".stripMargin.replaceAll("\n", " ")).select(col("sum(CAST(item_cnt_month AS DOUBLE))"))
              .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble / (numberOfShops - 1)
            val priceRelationalCoefficient = productPrice / meanPrice
            var productSalesModifified = 0
            var meanSalesModifified = 0
            if (productSales == 0.0) {
              productSales = 1
              productSalesModifified = 1
            }
            if (meanSales == 0.0) {
              meanSales = 1
              meanSalesModifified = 1
            }
            val salesRelationalCoefficient = productSales / meanSales
            if (productSalesModifified == 1) {
              productSales = 0
            }
            if (meanSalesModifified == 1) {
              meanSales = 0
            }

            if (counter == 0) {
              observation = shop.toString +: observation
              observation = item.toString +: observation
              observation = productSales.toString +: observation
              observation = productPrice.toString +: observation
              observation = relativeShopSize.toString +: observation // relativeShopSize
            }
            else if (counter != (windowExtension - 1)) {
              observation = productSales.toString +: observation
              observation = productPrice.toString +: observation
              observation = salesRelationalCoefficient.toString +: observation
              observation = priceRelationalCoefficient.toString +: observation
            }
            else if (counter == (windowExtension - 1) & month > windowExtension) {
              observation = productSales.toString +: observation
              observation = productPrice.toString +: observation
              observation = salesRelationalCoefficient.toString +: observation
              observation = priceRelationalCoefficient.toString +: observation
              counter2 += 1
              if (counter2 == 1) {
                structuredDataTest += observation.reverse
              }
              else {
                structuredDataTrain += observation.reverse
              }
              observation = Array[String]()
              observation = shop.toString +: observation
              observation = item.toString +: observation
              observation = productSales.toString +: observation
              observation = productPrice.toString +: observation
              observation = relativeShopSize.toString +: observation // relativeShopSize
              counter = 0
            }
            else if (counter == (windowExtension - 1) & month <= windowExtension) {
              observation = productSales.toString +: observation
              observation = productPrice.toString +: observation
              observation = salesRelationalCoefficient.toString +: observation
              observation = priceRelationalCoefficient.toString +: observation
              structuredDataTrain += observation.reverse
              observation = Array[String]()
              counter = -1
              counter2 = 0
              break
            }
          }
        }
      }
    }

    csvWriterTrain.writeAll(structuredDataTrain.toList)
    outputFileTrain.close()
    csvWriterTest.writeAll(structuredDataTest.toList)
    outputFileTest.close()
  }


  def getMeanShopSize(spark:SparkSession): Double ={
    val listOfShops = spark.sql(
      f"""SELECT DISTINCT(shop_id) FROM global_temp.shops
         |""".stripMargin.replaceAll("\n", " "))
    var shopSizes = 0
    var counter = 0
    for (row <- listOfShops.rdd.collect) {
      counter += 1
      val shopId = row(0)
      val targetShopSize = spark.sql(
        f"""SELECT SUM(item_cnt_month) FROM global_temp.deseasonalizedDf
           |WHERE shop_id = $shopId%s
           |""".stripMargin.replaceAll("\n", " ")).select(col("sum(CAST(item_cnt_month AS DOUBLE))"))
        .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble.toInt
      shopSizes += targetShopSize
    }
    val meanShopSize = shopSizes.toDouble / counter
    meanShopSize
  }


  def getRelativeShopSize(spark:SparkSession,shop:String,selectedShops:List[String],meanShopSize:Double): Double = {
    val targetShopSize = spark.sql(
      f"""SELECT SUM(item_cnt_month) FROM global_temp.deseasonalizedDf
         |WHERE shop_id = $shop%s
         |""".stripMargin.replaceAll("\n", " ")).select(col("sum(CAST(item_cnt_month AS DOUBLE))"))
      .first.toString().replace("[", "").replace("]", "").replace("null", "0").toDouble
    val relativeShopSize = targetShopSize / meanShopSize
    relativeShopSize
  }
}
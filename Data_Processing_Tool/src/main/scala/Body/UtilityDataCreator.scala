package Body

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import java.io._
import java.io.{BufferedWriter, FileWriter}
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer


object UtilityDataCreator extends Serializable {
  def selectShopsAndItems(spark:SparkSession,numberOfMonths:Double,numberOfShops:Double,numberOfItems:Int,numberOfShopsToUse:Int,numberOfItemsToUse:Int): (List[String],List[String]) = {
    println("Checking if the CSV with product prices exists...")
    val shopsBySales = spark.sql(
      f"""SELECT shop_id, SUM(item_cnt_day) FROM global_temp.sales_train
         |GROUP BY shop_id
         |ORDER BY length(SUM(item_cnt_day)) DESC, SUM(item_cnt_day) DESC
         |""".stripMargin.replaceAll("\n", " "))
    val itemsBySales = spark.sql(
      f"""SELECT item_id, SUM(item_cnt_day) FROM global_temp.sales_train
         |GROUP BY item_id
         |ORDER BY length(SUM(item_cnt_day)) DESC, SUM(item_cnt_day) DESC
         |""".stripMargin.replaceAll("\n", " "))
    var selectedShops = List[String]()
    var counter = 0
    for (shop <- shopsBySales.rdd.collect){
      counter += 1
      if (counter <= numberOfShopsToUse){
        selectedShops = selectedShops :+ shop(0).toString
      }
    }
    var selectedItems = List[String]()
    counter = 0
    for (item <- itemsBySales.rdd.collect){
      counter += 1
      if (counter <= numberOfItemsToUse){
        selectedItems = selectedItems :+ item(0).toString
      }
    }
    (selectedShops,selectedItems)
  }


  def createPricesView(spark:SparkSession,numberOfMonths:Double,numberOfShopsToUse:Int,numberOfItemsToUse:Int,selectedShops:List[String],selectedItems:List[String]): Unit = {
    var selectedShopsString = selectedShops.head.toString
    var counter = 0
    for (shop <- selectedShops){
      counter += 1
      if (counter != 1){
        selectedShopsString += "," + shop.toString
      }
    }
    var selectedItemsString = selectedItems.head.toString
    counter = 0
    for (item <- selectedItems){
      counter += 1
      if (counter != 1){
        selectedItemsString += "," + item.toString
      }
    }
    val monthly_prices = spark.sql(
      f"""SELECT shop_id, item_id, date_block_num, AVG(item_price) AS item_price FROM global_temp.sales_train
         |WHERE shop_id IN ($selectedShopsString%s)
         |AND item_id IN ($selectedItemsString%s)
         |GROUP BY shop_id, item_id, date_block_num
         |""".stripMargin.replaceAll("\n", " "))
    monthly_prices.createOrReplaceGlobalTempView("monthly_prices")

    var fittingProductPricesFileExists = false
    var fittingProductPricesFile = ""
    var previousFileSize = 0
    val d = new File("../Files/Product_prices")
    if (d.exists && d.isDirectory) {
      val existingProductPricesFiles = d.listFiles.filter(_.isFile).toList
      for (file <- existingProductPricesFiles){
        val fileParameters = file.toString.split("product_prices_").last.split(".csv")(0).split("_")
        if (fileParameters(0).toInt >= numberOfShopsToUse && fileParameters(1).toInt >= numberOfItemsToUse){
          val fileSize = fileParameters(0).toInt * fileParameters(1).toInt
          if (!fittingProductPricesFileExists || fileSize < previousFileSize){
            fittingProductPricesFileExists = true
            fittingProductPricesFile = file.toString
            previousFileSize = fileSize
          }
        }
      }
    }

    if (!fittingProductPricesFileExists){
      println("No fitting files found")
      val outputFilePrices = new BufferedWriter(new FileWriter(f"../Files/Product_prices/product_prices_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv"))
      val csvWriterPrices = new CSVWriter(outputFilePrices)
      var csvFields3 = Array("item_id", "date_block_num", "mean_product_price")
      var meanProductPrices = new ListBuffer[Array[String]]()
      meanProductPrices += csvFields3
      var line = Array[String]()

      counter = 0
      for (item <- selectedItems){
        counter += 1
        println(" ")
        if (counter == 1){
          println("Creating CSV file with product prices:")
        }
        println("Execution number: " + counter.toString)
        for (month <- 0 until numberOfMonths.toInt){
          print(month.toString + "  ")
          var meanProductPrice = spark.sql(
            f"""SELECT AVG(item_price) FROM global_temp.sales_train
               |WHERE item_id = $item%s
               |AND date_block_num = $month%s
               |""".stripMargin.replaceAll("\n", " ")).select(col("avg(CAST(item_price AS DOUBLE))"))
            .first.toString().replace("[","").replace("]","").replace("null","0")
          if (meanProductPrice == "0"){
            meanProductPrice = spark.sql(
              f"""SELECT AVG(item_price) FROM global_temp.sales_train
                 |WHERE item_id = $item%s
                 |""".stripMargin.replaceAll("\n", " ")).select(col("avg(CAST(item_price AS DOUBLE))"))
              .first.toString().replace("[","").replace("]","").replace("null","0")
          }
          line = line :+ item.toString
          line = line :+ month.toString
          line = line :+ meanProductPrice
          meanProductPrices += line
          line = Array[String]()
        }
      }

      println()
      csvWriterPrices.writeAll(meanProductPrices.toList)
      outputFilePrices.close()
    }
    else{
      println("The following fitting file have been found: " + fittingProductPricesFile)
    }

    if (fittingProductPricesFileExists){
      val product_prices = spark.read.format("csv").option("header", "true").load(f"$fittingProductPricesFile%s")
      product_prices.createOrReplaceGlobalTempView("product_prices")
    }
    else{
      val product_prices = spark.read.format("csv").option("header", "true").load(f"../Files/Product_prices/product_prices_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv")
      product_prices.createOrReplaceGlobalTempView("product_prices")
    }
  }
}
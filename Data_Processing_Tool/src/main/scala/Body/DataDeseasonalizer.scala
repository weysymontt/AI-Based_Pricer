package Body

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import scala.util.control.Breaks._
import java.io._
import java.io.{BufferedWriter, FileWriter}
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer


object DataDeseasonalizer {
  def deseasonalizeTheData(spark:SparkSession,numberOfMonths:Double,numberOfShopsToUse:Int,numberOfItemsToUse:Int,selectedShops:List[String],selectedItems:List[String]): Unit = {
    println()
    println("Checking if the CSV with deseasonalized data exists...")
    var fittingDeseasonalizedFileExists = false
    var fittingDeseasonalizedFile = ""
    var previousFileSize = 0
    val d = new File("../Files/Deseasonalized")
    if (d.exists && d.isDirectory) {
      val existingDeseasonalizedFiles = d.listFiles.filter(_.isFile).toList
      for (file <- existingDeseasonalizedFiles){
        val fileParameters = file.toString.split("deseasonalizedSales_").last.split(".csv")(0).split("_")
        if (fileParameters(0).toInt >= numberOfShopsToUse && fileParameters(1).toInt >= numberOfItemsToUse){
          val fileSize = fileParameters(0).toInt * fileParameters(1).toInt
          if (!fittingDeseasonalizedFileExists || fileSize < previousFileSize){
            fittingDeseasonalizedFileExists = true
            fittingDeseasonalizedFile = file.toString
            previousFileSize = fileSize
          }
        }
      }
    }

    if (!fittingDeseasonalizedFileExists){
      println("No fitting files found")
      println()
      println("Creating CSV with deseasonalized data:")

      val outputFile = new BufferedWriter(new FileWriter(f"../Files/Deseasonalized/deseasonalizedSales_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv"))
      val csvWriter = new CSVWriter(outputFile)
      val csvFields2 = Array("shop_id","item_id","date_block_num","item_price","item_cnt_month","seasonal_index")
      var shop_id = List[String]()
      var item_id = List[String]()
      var month_number = List[String]()
      var price = List[String]()
      var deseasonalized_item_sales = List[String]()
      var seasonal_index = List[String]()

      var temp_counter = 0
      for (shop <- selectedShops){
        for (item <- selectedItems){
          temp_counter += 1
          println("Execution number: " + temp_counter.toString)
          val item_price_dataframe = spark.sql(
            f"""SELECT date_block_num,item_price FROM global_temp.monthly_prices
               |WHERE item_id = $item%s
               |AND shop_id = $shop%s
               |""".stripMargin.replaceAll("\n", " ")).rdd.collect
          val item_price_altenative = spark.sql(
            f"""SELECT AVG(item_price) FROM global_temp.monthly_prices
               |WHERE item_id = $item%s
               |AND shop_id = $shop%s
               |""".stripMargin.replaceAll("\n", " ")).select(col("avg(item_price)"))
            .first.toString().replace("[","").replace("]","").replace("null","0.0").toDouble
          var item_price = 0.0
          var current_month = -1
          for (month <- 0 to numberOfMonths.toInt){
            for (row <- item_price_dataframe){
              if (month > current_month){
                val register_month = row.mkString(",").split(",")(0).toInt
                val value = row.mkString(",").split(",")(1).toDouble
                if (month == register_month){
                  item_price = value
                  current_month = register_month
                }
                else{
                  item_price = item_price_altenative
                }
              }
            }
            price = price :+ item_price.toString
          }

          val augmentedSales = getAugmentedSales(spark,shop,item,numberOfMonths)
          val totalSales = getTotalSales(augmentedSales,numberOfMonths)
          val totalSeasonalIndexes = getTotalSeasonalIndexes(totalSales,numberOfMonths)
          val meanSeasonalIndexes = getMeanSeasonalIndexes(totalSeasonalIndexes)
          val getDeseasonalizedSalesOutput = getDeseasonalizedSales(totalSales,meanSeasonalIndexes)
          val deseasonalizedSales = getDeseasonalizedSalesOutput._1
          val local_seasonal_index = getDeseasonalizedSalesOutput._2
          for (index <- local_seasonal_index){
            seasonal_index = seasonal_index :+ index
          }

          var counter = -1
          for (month <- deseasonalizedSales){
            counter += 1
            shop_id = shop_id :+ shop.toString
            item_id = item_id :+ item.toString
            month_number = month_number :+ counter.toString
            deseasonalized_item_sales = deseasonalized_item_sales :+ month.toString
          }
        }
      }

      var listOfRecords = new ListBuffer[Array[String]]()
      listOfRecords += csvFields2
      var counter = -1
      for (id <- item_id){
        counter += 1
        listOfRecords += Array(shop_id(counter), id, month_number(counter), price(counter), deseasonalized_item_sales(counter), seasonal_index(counter))
      }

      csvWriter.writeAll(listOfRecords.toList)
      outputFile.close()
    }
    else{
      println("The following fitting file have been found: " + fittingDeseasonalizedFile)
    }

    if (fittingDeseasonalizedFileExists){
      val deseasonalizedDf = spark.read.format("csv").option("header", "true").load(f"$fittingDeseasonalizedFile%s")
      deseasonalizedDf.createOrReplaceGlobalTempView("deseasonalizedDf")
    }
    else{
      val deseasonalizedDf = spark.read.format("csv").option("header", "true").load(f"../Files/Deseasonalized/deseasonalizedSales_$numberOfShopsToUse%s_$numberOfItemsToUse%s.csv")
      deseasonalizedDf.createOrReplaceGlobalTempView("deseasonalizedDf")
    }
  }


  def getAugmentedSales(spark:SparkSession,shop:String,item:String,numberOfMonths:Double): List[Int] = {
    val salesOfItem = spark.sql(
      f"""SELECT date_block_num,SUM(item_cnt_day) FROM global_temp.sales_train
         |WHERE shop_id = $shop%s
         |AND item_id = $item%s
         |GROUP BY date_block_num
         |ORDER BY length(date_block_num), date_block_num
         |""".stripMargin.replaceAll("\n", " "))

    var augmentedSales = List[Int]()
    var counter = -1
    for (row <- salesOfItem.rdd.collect){
      val month = row.mkString(",").split(",")(0)
      val sales = row.mkString(",").split(",")(1)
      breakable{
        for (i <- 0 until (numberOfMonths.toInt-1)){
          counter += 1
          if (month.toInt != counter){
            augmentedSales = augmentedSales :+ 0
          }
          else{
            augmentedSales = augmentedSales :+ sales.toDouble.toInt
            break()
          }
        }
      }
    }
    for (i <- counter to (numberOfMonths.toInt-2)){
      augmentedSales = augmentedSales :+ 0
    }
    augmentedSales
  }


  def getTotalSales(augmentedSales:List[Int],numberOfMonths:Double): List[List[String]] = {
    var yearCounter = 0
    var totalSales = List[List[String]]()
    var salesOverYear = List[String]()
    for (month <- augmentedSales){
      yearCounter += 1
      if (yearCounter < 12){
        salesOverYear = salesOverYear :+ month.toString
      }
      else{
        salesOverYear = salesOverYear :+ month.toString
        totalSales = totalSales :+ salesOverYear
        salesOverYear = List[String]()
        yearCounter = 0
      }
    }
    if (numberOfMonths % 12 != 0){
      totalSales = totalSales :+ salesOverYear
    }
    totalSales
  }


  def getTotalSeasonalIndexes(totalSales:List[List[String]],numberOfMonths:Double): List[List[Double]] = {
    var totalSeasonalIndexes = List[List[Double]]()
    var seasonalIndexesOverYear = List[Double]()
    for (year <- totalSales){
      var sumOfSalesOverYear = 0
      var counter = 0
      for (month <- year){
        sumOfSalesOverYear = sumOfSalesOverYear + month.toDouble.toInt
        counter += 1
      }
      val meanOfTheYear = sumOfSalesOverYear.toDouble / counter.toDouble
      counter = 0
      for (month <- year) {
        counter += 1
        val seasonalIndexForMonth = month.toDouble / meanOfTheYear.toDouble
        if (counter < 12){
          if (meanOfTheYear != 0){
            seasonalIndexesOverYear = seasonalIndexesOverYear :+ seasonalIndexForMonth
          }
          else{
            seasonalIndexesOverYear = seasonalIndexesOverYear :+ 0.0
          }
        }
        else{
          if (meanOfTheYear != 0){
            seasonalIndexesOverYear = seasonalIndexesOverYear :+ seasonalIndexForMonth
          }
          else{
            seasonalIndexesOverYear = seasonalIndexesOverYear :+ 0.0
          }
          totalSeasonalIndexes = totalSeasonalIndexes :+ seasonalIndexesOverYear
          seasonalIndexesOverYear = List[Double]()
          counter = 0
        }
      }
    }
    if (numberOfMonths % 12 != 0){
      totalSeasonalIndexes = totalSeasonalIndexes :+ seasonalIndexesOverYear
    }
    totalSeasonalIndexes
  }


  def getMeanSeasonalIndexes(totalSeasonalIndexes:List[List[Double]]): List[Double] = {
    var meanSeasonalIndexes = List[Double]()
    var sumOfSeasonalIndexes = 0.0
    var counter = -1
    var counter2 = 0
    for (month <- totalSeasonalIndexes.head){
      counter += 1
      for (year <- totalSeasonalIndexes){
        var counter3 = -1
        for (month <- year){
          counter3 += 1
          if (counter == counter3){
            sumOfSeasonalIndexes = sumOfSeasonalIndexes + month.toDouble
            counter2 += 1
          }
        }
      }
      meanSeasonalIndexes = meanSeasonalIndexes :+ (sumOfSeasonalIndexes / counter2)
      counter2 = 0
      sumOfSeasonalIndexes = 0
    }
    meanSeasonalIndexes
  }


  def getDeseasonalizedSales(totalSales:List[List[String]],meanSeasonalIndexes:List[Double]): (List[Double],List[String]) = {
    var deseasonalizedSales = List[Double]()
    var seasonal_index = List[String]()
    for (year <- totalSales){
      var counter = -1
      var counter2 = -1
      for (month <- year){
        counter += 1
        counter2 += 1
        if (meanSeasonalIndexes(counter) != 0){
          deseasonalizedSales = deseasonalizedSales :+ (month.toString.toDouble / meanSeasonalIndexes(counter))
          seasonal_index = seasonal_index :+ meanSeasonalIndexes(counter).toString
        }
        else{
          deseasonalizedSales = deseasonalizedSales :+ 0.0
          seasonal_index = seasonal_index :+ "1.0"
        }
      }
    }
    (deseasonalizedSales,seasonal_index)
  }
}
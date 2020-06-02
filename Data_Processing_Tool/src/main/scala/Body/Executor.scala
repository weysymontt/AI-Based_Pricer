package Body

import java.io._


object Executor extends Serializable {
  def main(args: Array[String]): Unit = {
    print("\u001b[2J")
    var exit = false
    while (!exit){
      var callNext = false
      var correctInput = false
      var firstTry = true
      while (!correctInput){
        println("DATA PROCESSING TOOL")
        println()
        val d = new File("../Files/Completely_processed_data")
        if (d.exists && d.isDirectory) {
          println("The following files with already processed data have been found:")
          val existingProcessedFiles = d.listFiles.filter(_.isFile).toList
          for (file <- existingProcessedFiles){
            val fileName = "shaped_input" + file.toString.split("shaped_input").last
            println(fileName)
          }
        }
        println()
        println("Do you want to proceed and process the data creating new files?")
        println("[1] YES")
        println("[2] NO, EXIT")
        println()
        if (!firstTry && !correctInput){
          println("The introduced value does not match the index of any of the available options")
        }
        println("Please introduce the option number:")

        firstTry = false

        val userInput = scala.io.StdIn.readLine()
        if (userInput == "1"){
          correctInput = true
          firstTry = true
          print("\u001b[2J")
          callNext = true
        }
        else if (userInput == "2"){
          correctInput = true
          firstTry = true
          print("\u001b[2J")
          exit = true
        }
        else{
          correctInput = false
          print("\u001b[2J")
        }
      }


      if (callNext){
        println("DATA PROCESSING TOOL")
        println()
        println("Creating a Spark session and reading the input data...")

        callNext = false
        val parameters = SparkSessionCreator.createSparkSession()
        val spark = parameters._1
        val numberOfMonths = parameters._2
        val numberOfShops = parameters._3
        val numberOfItems = parameters._4

        print("\u001b[2J")

        var windowExtension = 0
        var numberOfShopsToUse = 0
        var numberOfItemsToUse = 0
        var correctInputs = 0
        var errorMessage = ""
        var backToMainMenu = false
        firstTry = true
        while (!backToMainMenu && !callNext){
          while (!backToMainMenu && correctInputs != 3){
            print("\u001b[2J")
            println("DATA PROCESSING TOOL")
            println()
            if (!firstTry){
              println(errorMessage)
            }
            if (correctInputs == 0){
              println("Please introduce a window extension or [0] to go back to the main menu:")
            }
            else if (correctInputs == 1){
              println("Please introduce the number of shops to use or [0] to go back to the main menu:")
            }
            else if (correctInputs == 2){
              println("Please introduce the number of items to use or [0] to go back to the main menu:")
            }

            firstTry = false

            val userInput = try{scala.io.StdIn.readInt()} catch{case ex: NumberFormatException => -1}
            if (userInput != -1 && userInput.toInt > 0){
              print("\u001b[2J")
              firstTry = true
              if (correctInputs == 0){
                if (userInput < numberOfMonths){
                  windowExtension = userInput
                  correctInputs += 1
                }
                else{
                  errorMessage = f"There are only ${numberOfMonths.toInt}%s months of data available"
                }
              }
              else if (correctInputs == 1){
                if (userInput < numberOfShops){
                  numberOfShopsToUse = userInput
                  correctInputs += 1
                }
                else{
                  errorMessage = f"There are only ${numberOfShops.toInt}%s shops available"
                }
              }
              else if (correctInputs == 2){
                if (userInput < numberOfItems){
                  numberOfItemsToUse = userInput
                  correctInputs += 1
                }
                else{
                  errorMessage = f"There are only $numberOfItems%s items available"
                }
              }
            }
            else if (userInput == 0){
              backToMainMenu = true
              print("\u001b[2J")
            }
            else{
              errorMessage = "The introduced value should be a positive integer"
            }
          }

          var trainMatches = false
          var trainMatchingFile = ""
          var testMatches = false
          var testMatchingFile = ""
          if (correctInputs == 3){
            val d = new File("../Files/Completely_processed_data")
            if (d.exists && d.isDirectory) {
              val existingProcessedFiles = d.listFiles.filter(_.isFile).toList
              for (file <- existingProcessedFiles){
                if (file.toString.contains("train")){
                  val trainParameters = file.toString.split("shaped_input_train_").last.split(".csv")(0).split("_")
                  if (trainParameters(0).toInt == windowExtension && trainParameters(1).toInt == numberOfShopsToUse && trainParameters(2).toInt == numberOfItemsToUse){
                    trainMatches = true
                    trainMatchingFile = "shaped_input" + file.toString.split("shaped_input").last
                  }
                }
                else if (file.toString.contains("test")){
                  val testParameters = file.toString.split("shaped_input_test_").last.split(".csv")(0).split("_")
                  if (testParameters(0).toInt == windowExtension && testParameters(1).toInt == numberOfShopsToUse && testParameters(2).toInt == numberOfItemsToUse){
                    testMatches = true
                    testMatchingFile = "shaped_input" + file.toString.split("shaped_input").last
                  }
                }
              }
            }
          }

          if (!backToMainMenu){
            if (!trainMatches && !testMatches){
              correctInput = false
              firstTry = true
              while (!correctInput){
                println("DATA PROCESSING TOOL")
                println()
                println("The selected parameters are:")
                println(f"Window extension: $windowExtension%s")
                println(f"Number of shops to use: $numberOfShopsToUse%s")
                println(f"Number of items to use: $numberOfItemsToUse%s")
                println()
                println("Do you want to proceed and process the data creating new files with the specified parameters")
                println("[1] YES")
                println("[2] NO")
                println()
                if (!firstTry && !correctInput){
                  println("The introduced value does not match the index of any of the available options")
                }
                println("Please introduce the option number:")

                firstTry = false

                val userInput = scala.io.StdIn.readLine()
                if (userInput == "1"){
                  correctInput = true
                  firstTry = true
                  callNext = true
                  print("\u001b[2J")
                }
                else if (userInput == "2") {
                  correctInput = true
                  firstTry = true
                  backToMainMenu = true
                  print("\u001b[2J")
                }
                else{
                  correctInput = false
                  print("\u001b[2J")
                }
              }
            }
            else{
              correctInput = false
              firstTry = true
              while (!correctInput){
                println("DATA PROCESSING TOOL")
                println()
                println("The following files with the data processed with the selected parameters have been found:")
                println(trainMatchingFile)
                println(testMatchingFile)
                println()
                println("Do you want to proceed? Any existing file with these parameters will be replaced")
                println("[1] YES")
                println("[2] NO. Introduce different parameters")
                println("[3] NO. Go back to the main menu")
                println()
                if (!firstTry && !correctInput){
                  println("The introduced value does not match the index of any of the available options")
                }
                println("Please introduce the option number:")

                firstTry = false

                val userInput = scala.io.StdIn.readLine()
                if (userInput == "1"){
                  correctInput = true
                  firstTry = true
                  callNext = true
                  print("\u001b[2J")
                }
                else if (userInput == "2") {
                  correctInput = true
                  firstTry = true
                  correctInputs = 0
                  print("\u001b[2J")
                }
                else if (userInput == "3") {
                  correctInput = true
                  firstTry = true
                  backToMainMenu = true
                  print("\u001b[2J")
                }
                else{
                  correctInput = false
                  print("\u001b[2J")
                }
              }
            }
          }
        }


        if (callNext){
          println("DATA PROCESSING TOOL")
          println()
          println("Execution log:")

          // Select shops and items for the Neural Network input
          val selectedData = UtilityDataCreator.selectShopsAndItems(spark,numberOfMonths,numberOfShops,numberOfItems,numberOfShopsToUse,numberOfItemsToUse)
          val selectedShops = selectedData._1
          val selectedItems = selectedData._2

          // Create DF for prices. Takes 140 seconds per product approx.
          UtilityDataCreator.createPricesView(spark,numberOfMonths,numberOfShopsToUse,numberOfItemsToUse,selectedShops,selectedItems)

          // Deseasonalize the data. Takes approximately 18 seconds per item.
          DataDeseasonalizer.deseasonalizeTheData(spark,numberOfMonths,numberOfShopsToUse,numberOfItemsToUse,selectedShops,selectedItems)

          // Create an input for the Neural Network. Takes approximately 7 seconds per item.
          ProcessedDataCreator.CreateANNinput(spark,windowExtension,numberOfMonths,numberOfShops,numberOfShopsToUse,numberOfItemsToUse,selectedShops,selectedItems)

          println()
          println("The proceess has finished successfully. Press 'Enter' to go back to the main menu.")
          scala.io.StdIn.readLine()
          print("\u001b[2J")
        }
      }
    }
  }
}
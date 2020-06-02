import Dependencies._

ThisBuild / scalaVersion     := "2.12.10"
ThisBuild / version          := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "Data_Processing_Tool",
    libraryDependencies += scalaTest % Test,
    libraryDependencies += sparkCore,
    libraryDependencies += sparkMLlib
  )


/*javaOptions in run ++= Seq(
  "-Dlog4j.debug=false",
  "-Dlog4j.configuration=log4j.properties")
outputStrategy := Some(StdoutOutput)*/


// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.

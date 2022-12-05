import org.apache.log4j.{Level, Logger}
import org.apache.spark._

import java.io.{BufferedWriter, File, FileWriter}


object App {
    def main(args: Array[String]): Unit = {
        System.setProperty("hadoop.home.dir", "c:/winutils/")
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)
        val conf = new SparkConf().setAppName("AccidentPredictor").setMaster("local[4]")
        val sc = new SparkContext(conf)

        val accidents = sc.textFile("US_Accidents_Dec21_updated.csv").map(_.split(',')).persist
        val file = new File("accidents.csv")
        val bw = new BufferedWriter(new FileWriter(file))
        //0, 1, 2, 4, 5, 25, 28
        val filtered = accidents.map(acc => {
            val aid = acc(0).split("A-")
            var id = "0"
            if(aid.length > 1) { id = aid(1) }
            var time = acc(2)
            val dt = time.split(" ")
            if(dt.length > 1) { time = dt(1) }
            val timeParts = time.split(":")
            if(timeParts.length == 3) { time = timeParts(0) }
            var vis = acc(25)
            if( vis == "") {vis = "0"}
            var precip = acc(28)
            if( precip == "") {precip = "0"}
            (id, acc(1), time, acc(4), acc(5), vis, precip)})
            .filter{case (aid: String, _, stt: String, _, _, _, _) => aid.toInt > 0 && stt.toInt > 0}
            .map{ case (aid: String, sev: String, stt: String, stLat: String,
                stLng: String, vis: String, precip: String) => (sev, (aid, stt, stLat, stLng, vis, precip))}.persist

        val minGroupSize = filtered.map{case (sev: String, (_, _, _, _, _, _)) => (checkDouble(sev), 1)}
            .reduceByKey((x, y) => x + y).map{
            case (sev: Double, count: Int) => (count, sev)}.sortByKey().take(1).take(1)(0)._1

        val sample1 = filtered.filter{case (sev: String, (_, _, _, _, _, _)) => checkDouble(sev) == 1.0 }.map{
            case (sev: String, (aid: String, stt: String, stLat: String, stLng: String, vis: String, precip: String)) =>
                (aid, sev, stt, stLat, stLng, vis, precip).productIterator.toArray.mkString(",")}
        sample1.take(minGroupSize).foreach(str => {
            bw.write(str)
            bw.write("\n")})
        val sample2 = filtered.filter{case (sev: String, (_, _, _, _, _, _)) => checkDouble(sev) == 2.0 }.map{
            case (sev: String, (aid: String, stt: String, stLat: String, stLng: String, vis: String, precip: String)) =>
                (aid, sev, stt, stLat, stLng, vis, precip).productIterator.toArray.mkString(",")}
        sample2.take(minGroupSize).foreach(str => {
            bw.write(str)
            bw.write("\n")})
        val sample3 = filtered.filter{case (sev: String, (_, _, _, _, _, _)) => checkDouble(sev) == 3.0 }.map{
            case (sev: String, (aid: String, stt: String, stLat: String, stLng: String, vis: String, precip: String)) =>
                (aid, sev, stt, stLat, stLng, vis, precip).productIterator.toArray.mkString(",")}
        sample3.take(minGroupSize).foreach(str => {
            bw.write(str)
            bw.write("\n")})
        val sample4 = filtered.filter{case (sev: String, (_, _, _, _, _, _)) => checkDouble(sev) == 4.0 }.map{
            case (sev: String, (aid: String, stt: String, stLat: String, stLng: String, vis: String, precip: String)) =>
                (aid, sev, stt, stLat, stLng, vis, precip).productIterator.toArray.mkString(",")}
        sample4.take(minGroupSize).foreach(str => {
            bw.write(str)
            bw.write("\n")})
        bw.close()

        val k = 20
        val set = sc.textFile("accidents.csv").persist
        val trainingSet = set
            .sample(withReplacement = false, .8, 1)
            .map(line => {
            val tokens = line.split(',')
            var accidentVector = List[Double]()
            accidentVector = List(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if(tokens.length > 6){
                accidentVector = List(
                    checkDouble(tokens(1)),//sev
                    checkDouble(tokens(2)),//startTime
                    checkDouble(tokens(3)),//startLat
                    checkDouble(tokens(4)),//startLang
                    checkDouble(tokens(5)),//visibility
                    checkDouble(tokens(6)))//precipitation
            }
            accidentVector
        }).filter(l => l.head > 0.0).map(vector => (vector.head, vector.tail)).persist
        println(trainingSet.count())

        val testingSet = set.sample(withReplacement = false, .01, 1).map(line => {
            val tokens = line.split(',')
            var accidentVector = List[Double]()
            accidentVector = List(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if(tokens.length > 6){
                accidentVector = List(
                    checkDouble(tokens(1)),//sev
                    checkDouble(tokens(0)),//id
                    checkDouble(tokens(2)),//startTime
                    checkDouble(tokens(3)),//startLat
                    checkDouble(tokens(4)),//startLng
                    checkDouble(tokens(5)),//visibility
                    checkDouble(tokens(6)))//precipitation
            }
            accidentVector
        }).filter(l => l.head > 0.0).map(vector => (vector.head, vector.tail)).persist
        testingSet.take(1).foreach(println)
        println(testingSet.count())

        var maxTimeAll = 0.0
        var maxLatAll = 0.0
        var maxLngAll = 0.0
        var maxVisAll = 0.0
        var maxPrecipAll = 0.0

        testingSet.collect.foreach(acc => {
            val distances = trainingSet.map{case (severity: Double, vector: List[Double]) => (
                severity,
                List(getTimeDistance(acc._2(1), vector.head) * 1.0, //change 1.0 multiplier to change weight
                    Math.abs(acc._2(2) - vector(1)) * 1.0,
                    Math.abs(acc._2(3) - vector(2)) * 1.0,
                    Math.abs(acc._2(4) - vector(3)) * 1.0,
                    Math.abs(acc._2(5) - vector(4)) * 1.0))}.persist
            val maxTime = distances.map{case (_, distanceVector: List[Double]) => distanceVector.head}.top(1)
            if (maxTime.head > maxTimeAll) { maxTimeAll = maxTime.head }
            val maxLat = distances.map{case (_, distanceVector: List[Double]) => distanceVector(1)}.top(1)
            if (maxLat.head > maxLatAll) { maxLatAll = maxLat.head }
            val maxLng = distances.map{case (_, distanceVector: List[Double]) => distanceVector(2)}.top(1)
            if (maxLng.head > maxLngAll) { maxLngAll = maxLng.head }
            val maxVis = distances.map{case (_, distanceVector: List[Double]) => distanceVector(3)}.top(1)
            if (maxVis.head > maxVisAll) { maxVisAll = maxVis.head }
            val maxPrecip = distances.map{case (_, distanceVector: List[Double]) => distanceVector(4)}.top(1)
            if (maxPrecip.head > maxPrecipAll) { maxPrecipAll = maxPrecip.head }
        })

        val predicted = new File("output.csv")
        val bwp = new BufferedWriter(new FileWriter(predicted))

        testingSet.collect.foreach(acc => {
            val distances = trainingSet.map{case (severity: Double, vector: List[Double]) => (
                severity,
                List(getTimeDistance(acc._2(1), vector.head) * 1.0, //change 1.0 multiplier to change weight
                    Math.abs(acc._2(2) - vector(1)) * 1.0,
                    Math.abs(acc._2(3) - vector(2)) * 1.0,
                    Math.abs(acc._2(4) - vector(3)) * 1.0,
                    Math.abs(acc._2(5) - vector(4)) * 1.0))}.persist

            val eucDistances = distances.map{case (severity: Double, distanceVector: List[Double]) => (
                Math.sqrt(
                    scala.math.pow(distanceVector.head / maxTimeAll, 2) +
                        scala.math.pow(distanceVector(1) / maxLatAll, 2) +
                        scala.math.pow(distanceVector(2) / maxLngAll, 2) +
                        scala.math.pow(distanceVector(3) / maxVisAll, 2) +
                        scala.math.pow(distanceVector(4) / maxPrecipAll, 2)),
                severity)}
            val topNDistSev = eucDistances.sortByKey(ascending = false).take(k).map{
                case (_, sev: Double) => (sev, 1)}
            val sev = sc.parallelize(topNDistSev).reduceByKey((x, y) => x + y).map{
                case (sev: Double, count: Int) => (count, sev)}.sortByKey(ascending = false).take(1)(0)._2
            bwp.write(acc._2.head.toString + "," + acc._1.toString + "," + sev + "\n")
        })
        val modeledOut = sc.textFile("output.csv")      // ID -> known, predicted
            .map(_.split(",")).map(x => (x(0), (x(1), x(2)))).persist
        val severities = List("1.0", "2.0", "3.0", "4.0")
        var recallSum = 0.0
        for (s <- severities) {
            val TP = modeledOut.filter(x => x._2._1 == s && x._2._2 == s).count()    // gets true positives (num guessed correctly)
            val TPFN = modeledOut.filter(x => x._2._1 == s).count()  // true positives + false negatives (num of known severity values)
            val severityRecall = TP * 1.0 / TPFN
            recallSum += severityRecall
        }
        println(recallSum * 1.0 / 4)   // average of all 4 severity recall values
    }

    def checkDouble(DoubleHuh: String): Double = {
        try {
            val doub = DoubleHuh.toDouble
            doub
        }
        catch { case e: Exception => 0.0}
    }


    def getTimeDistance(A: Double, B: Double) : Double = {
        var ret = Math.abs(A - B)
        if(A >= B + 12.0) {
            ret = 12.0 - (A - (B + 12.0))
        }
        if(B >= A + 12.0) {
            ret = 12.0 - (B - (A + 12.0))
        }
        ret
    }
}

package com.wangyin.spark;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.commons.lang.ArrayUtils;
import org.apache.thrift.TException;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TSSLTransportFactory.TSSLTransportParameters;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.*;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

public class ModelClient {

    public static SVMModel train(JavaRDD<LabeledPoint> training) {
        // Run training algorithm to build the model.
        int numIterations = 100;
        final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);
        // Clear the default threshold.
        model.clearThreshold();
        return model;
    }

    private static void test(JavaRDD<LabeledPoint> test,
                             ModelManager.Client client,
                             JavaSparkContext sc) throws TException {
        List<Tuple2<Object, Object>> lst = new ArrayList<Tuple2<Object, Object>>();
        List<LabeledPoint> points = test.collect();
        for (LabeledPoint point : points) {
            Double score = client.predict("svm",
                    Arrays.asList(ArrayUtils.toObject(point.features().toArray())));
            lst.add(new Tuple2<Object, Object>(score, point.label()));
        }
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = sc.parallelize(lst);
        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        System.out.println("Area under ROC = " + auROC);
    }

    public static void main(String [] args) throws IOException {
        try {
            TTransport transport;
            transport = new TSocket("localhost", 9090);
            transport.open();

            TProtocol protocol = new  TBinaryProtocol(transport);
            ModelManager.Client client = new ModelManager.Client(protocol);

            perform(client);
            transport.close();
        } catch (TException x) {
            x.printStackTrace();
        }
    }

    private static void perform(ModelManager.Client client) throws TException, IOException
    {
        client.ping();
        System.out.println("ping()");

        int sum = client.add(1,1);
        System.out.println("1+1=" + sum);

        SparkConf conf = new SparkConf().setAppName("SVM Classifier Example");
        conf.setMaster("local");
        JavaSparkContext sc = new  JavaSparkContext(conf);
        String path = "/data/mllib/sample_libsvm_data.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), path).toJavaRDD();

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = data.subtract(training);

        SVMModel model = train(training);
        ByteArrayOutputStream bOutput = new ByteArrayOutputStream();
        ObjectOutputStream oOutput=new ObjectOutputStream(bOutput);
        oOutput.writeObject(model);
        oOutput.close();

        client.uploadModel("svm", ByteBuffer.wrap(bOutput.toByteArray()),
                "org.apache.spark.mllib.classification.SVMModel", MLType.CLASSIFICATION);

        try {
            Thread.sleep(1000);
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        test(test, client, sc);
    }
}

package com.wangyin.spark;

import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.RegressionModel;
import org.apache.thrift.TException;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.ObjectInputStream;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.commons.lang.ArrayUtils;

import org.apache.spark.mllib.classification.ClassificationModel;

public class ModelManagerHandler implements ModelManager.Iface {

    public static class ModelMetaInfo {
        public ByteBuffer model = null;
        public String className = "";
        public MLType type = null;
    }

    Map<String, ModelMetaInfo> models = new HashMap<String, ModelMetaInfo>();

    @Override
    public void ping() throws TException {

    }

    @Override
    public int add(int num1, int num2) throws TException {
        return num1 + num2;
    }

    @Override
    public boolean uploadModel(String name, ByteBuffer model, String className, MLType type) throws TException {
        ModelMetaInfo modelMetaInfo = new ModelMetaInfo();
        modelMetaInfo.model = model;
        modelMetaInfo.className = className;
        modelMetaInfo.type = type;
        models.put(name, modelMetaInfo);
        System.out.println("upload model "+  modelMetaInfo.type + "\n" +
                modelMetaInfo.className + " success! size:  " + model.limit());
        return true;
    }

    @Override
    public boolean deleteModel(String name) throws TException {
        models.remove(name);
        return true;
    }

    @Override
    public double predict(String name, List<Double> features) throws TException {
        ModelMetaInfo modelMetaInfo = models.get(name);
        ObjectInputStream modelObject = null;
        double result = 0;
        Vector points = new DenseVector(ArrayUtils.toPrimitive(features.toArray(new Double[0])));
        try {
            modelObject = new ObjectInputStream(new ByteArrayInputStream(modelMetaInfo.model.array()));
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (modelObject != null) {
            Object model = null;
            try {
                model = modelObject.readObject();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
            Class clazz = null;
            try {
                clazz = Class.forName(modelMetaInfo.className);
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
            MLType type = modelMetaInfo.type;
            switch (type) {
                case CLASSIFICATION:
                    ClassificationModel clsModel = (ClassificationModel)model;
                    result = clsModel.predict(points);
                    break;
                case CLUSTERING:
                    KMeansModel clusterModel = (KMeansModel)model;
                    result = clusterModel.predict(points);
                    break;
                case REGRESSION:
                    RegressionModel regressionModel = (RegressionModel)model;
                    result = regressionModel.predict(points);
                    break;
                case RECOMMENDATION:

                    break;
            }
        }
        return result;
    }
}

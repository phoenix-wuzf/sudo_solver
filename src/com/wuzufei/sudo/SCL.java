package com.wuzufei.sudo;

import java.util.Date;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;

public class SCL {
    public static final int K = 5;

    public static void main(String[] args) {
        //Must: to load native opencv library (you must add the DLL to the library path first)
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Random random = new Random(new Date().getTime());

        //prepare trainData and trainLabel
        Mat trainData = new Mat(100, 2, CvType.CV_32FC1, new Scalar(1));
        Mat trainLabel = new Mat(100, 1, CvType.CV_32FC1, new Scalar(1));
        for (int i = 0; i < 50; i++) {
            trainData.put(i, 0, random.nextInt(100));
            trainData.put(i, 1, random.nextInt(100));
            trainLabel.put(i, 0, 0);
        }
        for (int i = 50; i < 100; i++) {
            trainData.put(i, 0, random.nextInt(100) + 100);
            trainData.put(i, 1, random.nextInt(100) + 100);
            trainLabel.put(i, 0, 1);
        }

        // System.out.println("trainData:\n" + trainData.dump());
        // System.out.println("trainLabel:\n" + trainLabel.dump());

        //train data using KNN
        //CvKNearest knn = new CvKNearest();
        KNearest knn1 = KNearest.create();
        boolean success = knn1.train(trainData, Ml.ROW_SAMPLE , trainLabel);
        System.out.println("training result: " + success);

        //prepare test data
        Mat testData = new Mat(100, 2, CvType.CV_32FC1, new Scalar(1));
        for (int i = 0; i < 100; i++) {
            int r = random.nextInt(200);
            testData.put(i, 0, r);
            testData.put(i, 1, r);
        }

        //find the nearest neighbours of test data
        Mat results = new Mat();
        Mat neighborResponses = new Mat();
        Mat dists = new Mat();
        knn1.findNearest(testData, K, results, neighborResponses, dists);
        //knn1.find_nearest(testData, K, results, neighborResponses, dists);

        // print out the results
        System.out.print("testData:\n" + testData.dump() + ": " + results.dump());
        System.out.println("results:\n" + results.dump());
        //System.out.println("neighborResponses:\n" + neighborResponses.dump());
        //System.out.println("dists:\n" + dists.dump());

    }
}
package com.wuzufei.sudo;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    public static void main(String[] args) {
	// write your code here

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat image = Imgcodecs.imread("C:\\Users\\phoenix-Wu\\Desktop\\test.jpg");
        Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
        Imgproc.adaptiveThreshold(image, image, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 25, 10);
        Imgcodecs.imwrite("C:\\Users\\phoenix-Wu\\Desktop\\test2.jpg", image);

        System.out.println("hello world\n");
        return;
    }
}

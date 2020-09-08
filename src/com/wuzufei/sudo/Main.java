package com.wuzufei.sudo;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
	// write your code here

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        //1 获取原图
        Mat src = Imgcodecs.imread("C:\\\\Users\\\\phoenix-Wu\\\\Desktop\\\\test.png");
        //2 图片灰度化
        Mat gary = new Mat();
        Imgproc.cvtColor(src, gary, Imgproc.COLOR_RGB2GRAY);
        //3 图像边缘处理
        Mat edges = new Mat();
        Imgproc.Canny(gary, edges, 200, 500, 3, false);
        //4 发现轮廓

        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, list, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        //5 绘制轮廓
        for (int i = 0, len = list.size(); i < len; i++) {
            if (i % 2 != 0) {
                Imgproc.drawContours(src, list, i, new Scalar(0, 255, 0), 1, Imgproc.LINE_AA);
            }
        }
        System.out.println(list.size());
        HighGui.imshow("111", src);
        HighGui.waitKey(0);


        return;
    }
}

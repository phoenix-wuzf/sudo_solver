package com.wuzufei.sudo;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.font.ImageGraphicAttribute;
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
        Imgproc.cvtColor(src, gary, Imgproc.COLOR_BGR2GRAY);
        //3 图像边缘处理
        Mat thresh = new Mat();
        Mat dilate = gary.clone();
        //Imgproc.Canny(gary, edges, 200, 500, 3, false);
        Imgproc.threshold(gary, thresh, 200, 255, 1);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(5,5));
        Imgproc.dilate(thresh, dilate, kernel);
        //4 发现轮廓

        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilate, list, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        int m = 0;
        for (int i = 0, len = list.size(); i < len; i++) {
            //System.out.println(hierarchy.get(0, i));
//            if ((hierarchy.get(0, i))[3] == 0 && (hierarchy.get(0, i))[2] > 0) {
//                m++;
//                int idx = (int) (hierarchy.get(0, i))[0];
//                Imgproc.drawContours(src, list, idx, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
//            }
            if ((hierarchy.get(0, i))[3] == 0) {
                if ((hierarchy.get(0, i))[2] > 0) {
                    m++;
                    int idx = (int) (hierarchy.get(0, i))[2];
                    //Imgproc.drawContours(src, list, i, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
                    Rect area = Imgproc.boundingRect(list.get(idx));
                    Imgproc.rectangle(src, area, new Scalar(0, 0, 255), 2);
                }
            }
        }
        System.out.println(list.size());
        System.out.println(m);
        HighGui.imshow("111", src);
        HighGui.waitKey(0);


        return;
    }
}

package com.wuzufei.sudo;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Main {
    public static void main(String[] args) {
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
        //HighGui.imshow("111", src);
        //HighGui.waitKey(0);
        process_tranin_data();

        return;
    }
    public static void process_tranin_data() {
        Mat train_collection = new Mat();
        List<Integer> train_label = new ArrayList<>();
        Mat test_collection = new Mat();
        List<Integer> test_label = new ArrayList<>();
        for (int i = 1; i < 11; i++) {
            String file_path = "D:\\work_space\\sudo_solver\\numbers\\"+i+".jpg";
            Mat src_image = Imgcodecs.imread(file_path);
            Mat gray = new Mat();
            Mat blur = new Mat();
            Mat thresh = new Mat();
            Imgproc.cvtColor(src_image, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
            Imgproc.adaptiveThreshold(blur, thresh, 255, 1, 1, 11, 2);
            Mat hierarchy = new Mat();
            List<MatOfPoint> list = new ArrayList<MatOfPoint>();
            Imgproc.findContours(thresh, list, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            int height = src_image.height();
            int width = src_image.width();
            List<Rect> list_1 = new ArrayList<>();
            List<Rect> list_2 = new ArrayList<>();
            for (int j = 0; j < list.size(); j++) {
                Rect rect = Imgproc.boundingRect(list.get(j));
                if (rect.height < 60 || rect.width < 30) {
                    continue;
                }
                if (rect.width > 30 && rect.height > (height / 4)) {
                    if (rect.y < (height / 2)) {
                        list_1.add(rect);
                        Imgproc.rectangle(src_image, rect, new Scalar(0, 0, 255), 2);
                    } else {
                        list_2.add(rect);
                        Imgproc.rectangle(src_image, rect, new Scalar(0, 255, 0), 2);
                    }
                }
                list_1.sort(new Comparator<Rect>() {
                    @Override
                    public int compare(Rect o1, Rect o2) {
                        return o1.x - o2.x;
                    }
                });
                list_2.sort(new Comparator<Rect>() {
                    @Override
                    public int compare(Rect o1, Rect o2) {
                        return o1.x - o2.x;
                    }
                });
            }


            for (int j = 0; j < 5; j++) {
                String file_out_path = "D:\\work_space\\sudo_solver\\train_data\\"+ (j + 1) +"\\" + ((i + 1) *(j + 1)) +".jpg";
                Mat roi_1 = new Mat(thresh, list_1.get(j));
                Imgproc.resize(roi_1, roi_1, new Size(40, 80));
                //Imgcodecs.imwrite(file_out_path, roi_1);
                if (i == 3) {
                    test_collection.push_back(roi_1);
                    test_label.add(j+1);
                    continue;
                }
                train_collection.push_back(roi_1);
                train_label.add(j+1);

            }
            for (int j = 0; j < 4; j++) {
                String file_out_path = "D:\\work_space\\sudo_solver\\train_data\\"+ (j + 6) +"\\" + ((i + 1) *(j + 1)) +".jpg";
                Mat roi_2 = new Mat(thresh, list_2.get(j));
                Imgproc.resize(roi_2, roi_2, new Size(40, 80));
                //Imgcodecs.imwrite(file_out_path, roi_2);
                if (i == 3) {
                    test_collection.push_back(roi_2);
                    test_label.add(j+1);
                    continue;
                }
                train_collection.push_back(roi_2);
                train_label.add(j+6);
            }

            HighGui.imshow(i + "jpg", src_image);
            HighGui.waitKey(0);

        }
        KNearest knn = KNearest.create();

        boolean status =  knn.train(train_collection, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(train_label));
        System.out.println("status:" + status);
    }
}

package com.wuzufei.sudo;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import javax.swing.*;
import java.sql.SQLSyntaxErrorException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class integreateTest {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        JFileChooser fileChooser = new JFileChooser("D:\\");
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int returnVal = fileChooser.showOpenDialog(fileChooser);
        String filePath = "";
        if(returnVal == JFileChooser.APPROVE_OPTION) {
            filePath = fileChooser.getSelectedFile().getAbsolutePath();//这个就是你选择的文件夹的路径
        }
        //1 获取原图
        //Mat src = Imgcodecs.imread("C:\\\\Users\\\\phoenix-Wu\\\\Desktop\\\\test2.png");
        Mat src = Imgcodecs.imread(filePath);
        System.out.println(src.width() + " " + src.height());
        Imgproc.resize(src, src, new Size(952, 952));
        //2 图片灰度化
        Mat gary = new Mat();
        Imgproc.cvtColor(src, gary, Imgproc.COLOR_BGR2GRAY);
        //3 图像边缘处理
        Mat thresh = new Mat();
        Mat dilate = gary.clone();
        //Imgproc.Canny(gary, edges, 200, 500, 3, false);
        Imgproc.threshold(gary, thresh, 200, 255, 1);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(5, 5));
        Imgproc.dilate(thresh, dilate, kernel);
        //4 发现轮廓

        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Mat test_img_data = new Mat();
        Imgproc.findContours(dilate, list, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        int row = 0, col = 0, cnt = 0;
        char[][] sudo_img = new char[10][10];

        //boolean status = knn_train();
        KNearest knn = knn_train();
        for (int i = list.size() - 1; i >= 0; i--) {
            if ((hierarchy.get(0, i))[3] == 0) {
                row = cnt / 9;
                col = cnt % 9;
                cnt++;
                if ((hierarchy.get(0, i))[2] > 0) {
                    int idx = (int) (hierarchy.get(0, i))[2];
                    Rect area = Imgproc.boundingRect(list.get(idx));
                    Mat roi = new Mat(thresh, area);
                    Imgproc.resize(roi, roi, new Size(40, 80));
                    roi.convertTo(roi, CvType.CV_32F);
                    test_img_data.push_back(roi.reshape(1, 1));

                    int find_num = (int) knn.findNearest(roi.reshape(1, 1), 1, new Mat());
                    System.out.println("num[" + row + "][" + col + "]:" + find_num);
                    sudo_img[row][col] = (char) (find_num + '0');
                    continue;
                }
                sudo_img[row][col] = '.';
            }
        }
        for (int i = 0; i < sudo_img.length; i++) {
            for (int j = 0; j < sudo_img[0].length; j++) {
                System.out.print(sudo_img[i][j] + " ");
            }
            System.out.println();
        }
        SudoSolver sudo_solver = new SudoSolver();
        sudo_solver.solveSudoku(sudo_img);
        System.out.println();
        for (int i = 0; i < sudo_img.length; i++) {
            for (int j = 0; j < sudo_img[0].length; j++) {
                System.out.print(sudo_img[i][j] + " ");
            }
            System.out.println();
        }
        cnt = 0;
        for (int i = list.size() - 1; i >= 0; i--) {
            if ((hierarchy.get(0, i))[3] == 0) {
                row = cnt / 9;
                col = cnt % 9;
                cnt++;
                if ((hierarchy.get(0, i))[2] > 0) {
                    int idx = (int) (hierarchy.get(0, i))[2];
                    Rect area = Imgproc.boundingRect(list.get(idx));
                } else {
                    Rect area = Imgproc.boundingRect(list.get(i));
                    Imgproc.putText(src, String.valueOf(sudo_img[row][col]), new Point(area.x + 20, area.y + 70),
                            3, 2.5, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
                }

            }
        }
        HighGui.imshow("111", src);
        HighGui.waitKey(0);
        return;
    }

    public static KNearest knn_train() {
        boolean status;
        Mat train_collection = new Mat();
        List<Integer> train_label = new ArrayList<>();
        for (int i = 1; i < 11; i++) {
            String file_path = "D:\\work_space\\sudo_solver\\numbers\\" + i + ".jpg";
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
                Mat roi_1 = new Mat(thresh, list_1.get(j));
                Imgproc.resize(roi_1, roi_1, new Size(40, 80));
                roi_1.convertTo(roi_1, CvType.CV_32F);
                train_collection.push_back(roi_1.reshape(1, 1));
                train_label.add(j + 1);
            }
            for (int j = 0; j < 5; j++) {
                Mat roi_2 = new Mat(thresh, list_2.get(j));
                Imgproc.resize(roi_2, roi_2, new Size(40, 80));
                roi_2.convertTo(roi_2, CvType.CV_32F);
                train_collection.push_back(roi_2.reshape(1, 1));
                train_label.add(j + 6);
            }
        }
        KNearest knn = KNearest.create();
        status = knn.train(train_collection, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(train_label));

        return knn;
    }
}
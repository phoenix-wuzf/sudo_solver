package com.wuzufei.sudo;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;
import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class integreateTest {
    private static String train_data_path = "D:\\work_space\\sudo_solver\\numbers\\";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        JFileChooser fileChooser = new JFileChooser("D:\\");
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int returnVal = fileChooser.showOpenDialog(fileChooser);
        String filePath = "";
        if(returnVal == JFileChooser.APPROVE_OPTION) {
            filePath = fileChooser.getSelectedFile().getAbsolutePath();
        }
        //1 获取原图
        Mat src = Imgcodecs.imread(filePath);
        Imgproc.resize(src, src, new Size(952, 952));
        //2 图片灰度化
        Mat gary = new Mat();
        Imgproc.cvtColor(src, gary, Imgproc.COLOR_BGR2GRAY);
        //3 图像边缘处理
        Mat thresh = new Mat();
        Mat dilate = gary.clone();
        Imgproc.threshold(gary, thresh, 200, 255, 1);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(5, 5));
        Imgproc.dilate(thresh, dilate, kernel);
        //4 发现轮廓
        List<MatOfPoint> list = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilate, list, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        int row = 0, col = 0, cnt = 0;
        char[][] sudo_img = new char[10][10];

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

                    int find_num = (int) knn.findNearest(roi.reshape(1, 1), 1, new Mat());
                    sudo_img[row][col] = (char) (find_num + '0');
                    continue;
                }
                sudo_img[row][col] = '.';
            }
        }

        SudoSolver sudo_solver = new SudoSolver();
        sudo_solver.solveSudoku(sudo_img);

        /* 识别结果添加到原图中 */
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

        Imgcodecs.imwrite(filePath.split("\\.")[0] + "_result.png", src);
    }

    private static KNearest knn_train() {
        boolean status;
        Mat train_collection = new Mat();
        List<Integer> train_label = new ArrayList<>();
        for (int i = 1; i < 11; i++) {

            Mat src_image = Imgcodecs.imread(train_data_path + i + ".jpg");
            Mat tmp_mat = new Mat();
            /* 训练数据预处理 */
            Imgproc.cvtColor(src_image, tmp_mat, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(tmp_mat, tmp_mat, new Size(5, 5), 0);
            Imgproc.adaptiveThreshold(tmp_mat, tmp_mat, 255, 1, 1, 11, 2);

            Mat hierarchy = new Mat();
            List<MatOfPoint> list = new ArrayList<MatOfPoint>();
            Imgproc.findContours(tmp_mat, list, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            int height = src_image.height();
            List<Rect> list_1 = new ArrayList<Rect>();
            List<Rect> list_2 = new ArrayList<Rect>();

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
            }
            list_1.sort((o1, o2) -> (o1.x - o2.x));
            list_2.sort((o1, o2) -> (o1.x - o2.x));
            list_1.addAll(list_2);
            for (int j = 0; j < list_1.size(); j++) {
                Mat roi_1 = new Mat(tmp_mat, list_1.get(j));
                Imgproc.resize(roi_1, roi_1, new Size(40, 80));
                roi_1.convertTo(roi_1, CvType.CV_32F);
                train_collection.push_back(roi_1.reshape(1, 1));
                train_label.add(j + 1);
            }
        }
        KNearest knn = KNearest.create();
        status = knn.train(train_collection, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(train_label));
        if (status != true) {
            System.out.println("KNN is fail\n");
        }
        return knn;
    }
}
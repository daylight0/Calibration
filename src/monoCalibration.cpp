#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


int main( int argc, char** argv )
{
	if(argc != 5){
		cerr << "input: ./filename [image path] [boardwidth] [boardheight] [Square side length]" << endl;
		return -1;
	}

	// 读取一副图片，不改变图片本身的颜色类型（该读取方式为DOS运行模式）
	Mat image, grayImage;
	
	// 读取文件夹下文件的文件名
	vector<string> fileName;
	glob(argv[1], fileName);

	// 定义存储的容器
	vector<vector<Point3f>> objectPoints; // 棋盘上的空间坐标的真实点
	vector<vector<Point2f>> imagePoint;

	vector<Point2f> imageCornerBuf;
	vector<Point3f> objectCorner;

	Size imageSize;

	int imageCount = 0;

	// 输出参数定义
	Mat cameraMatrix; //输出内参数矩阵
	Mat distCoeffs;   //输出畸变矩阵

	vector<Mat> rvecs; //　旋转矩阵
	vector<Mat> tvecs; //　平移向量

	// 评价变量
	double totalError = 0.0;
	double error = 0.0;
	vector<Point2f> pixelPoint;


	//	计算棋盘上的真实点在3维空间上的坐标
	for (int i = 0; i < atoi(argv[3]); i++)
    {
       for (int j = 0; j < atoi(argv[2]); j++)
       {
		   Point3f tempPoint;
		   tempPoint.x = i * 29;
		   tempPoint.y = j * 29;
		   tempPoint.z = 0;
           objectCorner.push_back(tempPoint);
       }
    }

	// 提取棋盘上的角点
	cout << "Start extracting chessboard corners.\n"
			<< "Please wait.......\n";
	for(size_t i = 0; i < fileName.size(); ++i){
		image = imread(fileName[i], IMREAD_COLOR);
		if(image.empty()){
			cerr << "warning: there is an empty image!\n";
			cerr << fileName[i] << endl;
			continue;
		}
		cvtColor(image, grayImage, COLOR_BGR2GRAY);
		bool flag = findChessboardCorners(image, Size(atoi(argv[2]), atoi(argv[3])), imageCornerBuf, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+
										  CALIB_CB_FAST_CHECK);
		if(!flag){
			cout << "can not find chessboard corners!\n";
		    cout << "error: " << fileName[i] << endl;
			return -1;
		}else{
			//指定亚像素计算迭代标注
			TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.001);
			//亚像素检测
			cv::cornerSubPix(grayImage, imageCornerBuf, cv::Size(5, 5), cv::Size(-1, -1), criteria);
			// 判断提取的点和所设置的数目是否相同
			if(imageCornerBuf.size() == atoi(argv[2]) * atoi(argv[3])){
				imagePoint.push_back(imageCornerBuf);
				objectPoints.push_back(objectCorner);
			}
			drawChessboardCorners(image, Size(6, 9), imageCornerBuf, flag);
			imshow("Corners on Chessboard", image);
            waitKey(100);
		}
		imageSize = image.size();
		++imageCount;
	}
	destroyAllWindows();

	cout << "Start Calibration, Please waitting...\n";
	calibrateCamera(objectPoints,   // 三维点
                    imagePoint,     // 图像点
                    imageSize,      // 图像尺寸
                    cameraMatrix,   // 输出相机矩阵
                    distCoeffs,     // 输出畸变矩阵
                    rvecs, tvecs    // Rs、Ts（外参）
            );

	cout << "K = \n" << cameraMatrix << endl;
    cout << "distCoeffs = \n" << distCoeffs << endl << endl;


	// 对标定结果评价
	cout << "\n\nEvaluate the results of the correction parameters!\n";

    // clear the old xxx.yaml file. and save result
	string yamlFileOutPath = string(argv[1]) + "calibration.yaml";
	ofstream yamlFile(yamlFileOutPath);
	yamlFile.clear();

	yamlFile << imageCount << " Photos in total.\n";
	for(size_t i = 0; i < fileName.size(); ++i){

		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, pixelPoint);

		vector<Point2f> tempImagePoint = imagePoint[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat  = Mat(1, pixelPoint.size(), CV_32FC2);
		for ( int j = 0; j < tempImagePoint.size(); ++j){
			image_points2Mat.at<Vec2f>(0,j) = Vec2f(pixelPoint[j].x, pixelPoint[j].y);
			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		error = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		error /=  (atoi(argv[2]) * atoi(argv[3]));   
		totalError += error;
		yamlFile << " \nAverage error for the " << (i + 1) << "th image: " << error << " pixel";
	}
	yamlFile << "\nOverall mean error: " << totalError / imageCount << " pixel \n\n\n";

    yamlFile << "# ------camera Intrinsic--------" << endl;
    yamlFile << "Camera.fx:  " << cameraMatrix.at<double>(0,0) << endl;
    yamlFile << "Camera.fy:  " << cameraMatrix.at<double>(1,1) << endl;
    yamlFile << "Camera.cx:  " << cameraMatrix.at<double>(0,2) << endl;
    yamlFile << "Camera.cy:  " << cameraMatrix.at<double>(1,2) << endl;

    yamlFile << "\n# ------camera Distortion--------"<<endl;
    yamlFile << "Camera.k1:  " << distCoeffs.at<double>(0,0) << endl;
    yamlFile << "Camera.k2:  " << distCoeffs.at<double>(0,1) << endl;
    yamlFile << "Camera.p1:  " << distCoeffs.at<double>(0,2) << endl;
    yamlFile << "Camera.p2:  " << distCoeffs.at<double>(0,3) << endl;
    yamlFile << "Camera.k3:  " << distCoeffs.at<double>(0,4) << endl;

    yamlFile.close();

	FileStorage yamlf(yamlFileOutPath, FileStorage::APPEND);
    yamlf.writeComment(" \n------camera Intrinsic saved by yaml data--------",true);
    yamlf << "camera Matrix K" << cameraMatrix;
    yamlf.writeComment(" \n------camera Distortion saved by yaml data--------",true);
    yamlf << "camera Distortion" << distCoeffs;
    	
	yamlf.release();
 
	cout << "Correction parameters are successfully obtained and stored, storage path in " << argv[1] << endl;
	return 0;
}






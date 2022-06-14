#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

const int ZERO = 0;
const int NEGATIVE = -1;
const int TWO = 2;


int cameraCalibrate(vector<string> &files,  int height, int width, float squareSize, Mat &cameraMatrix, Mat &distCoeffs,
                    vector<vector<Point2f>> &imagePoint, Size &imageSize, vector<vector<Point3f>> &objectPoints,
                    vector<Mat> &rvecs, vector<Mat> &tvecs)
{

    Mat image, grayImage;

    vector<Point3f> objectCorner;
    vector<Point2f> imageCornerBuf;

     //	计算棋盘上的真实点在3维空间上的坐标
	for (int i = 0; i < height; i++)
    {
       for (int j = 0; j < width; j++)
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
	for(size_t i = 0; i < files.size(); ++i){
		image = imread(files[i], IMREAD_COLOR);
		if(image.empty()){
			cerr << "warning: there is an empty image!\n";
			cerr << files[i] << endl;
			continue;
		}
		cvtColor(image, grayImage, COLOR_BGR2GRAY);
		bool flag = findChessboardCorners(image, Size(width, height), imageCornerBuf, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+
										  CALIB_CB_FAST_CHECK);
		if(!flag){
			cout << "can not find chessboard corners!\n";
		    cout << "error: " << files[i] << endl;
			return -1;
		}else{
			//指定亚像素计算迭代标注
			TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 60, 0.001);
			//亚像素检测
			cv::cornerSubPix(grayImage, imageCornerBuf, cv::Size(5, 5), cv::Size(-1, -1), criteria);
			// 判断提取的点和所设置的数目是否相同
			if(imageCornerBuf.size() == (height * width)){
				imagePoint.push_back(imageCornerBuf);
				objectPoints.push_back(objectCorner);
			}
			drawChessboardCorners(image, Size(height, width), imageCornerBuf, flag);
			imshow("Corners on Chessboard", image);
            waitKey(100);
		}
		imageSize = image.size();
	}

	destroyAllWindows();

    cout << "Start Calibration, Please waitting...\n";
	double rms = calibrateCamera(objectPoints,   // 三维点
                    imagePoint,     // 图像点
                    imageSize,      // 图像尺寸
                    cameraMatrix,   // 输出相机矩阵
                    distCoeffs,     // 输出畸变矩阵
                    rvecs, tvecs    // Rs、Ts（外参）
            );

	cout << "K = \n" << cameraMatrix << endl;
    cout << "distCoeffs = \n" << distCoeffs << endl << endl;

    return 0;
}

// 冲投影误差函数
int reprojectionError(vector<string> &files, int height, int width, vector<vector<Point3f>> &objectPoints, vector<Mat> &rvecs, 
                     vector<Mat> &tvecs, Mat &cameraMatrix, Mat &distCoeffs, vector<Point2f> &pixelPoint, vector<vector<Point2f>> &imagePoint,
                     vector<double> &errorSave, double &totalError
            )
{
	double error = 0.0;

    for(size_t i = 0; i < files.size(); ++i){

		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, pixelPoint);

		vector<Point2f> tempImagePoint = imagePoint[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat  = Mat(1, pixelPoint.size(), CV_32FC2);
		for ( int j = 0; j < tempImagePoint.size(); ++j){
			image_points2Mat.at<Vec2f>(0,j) = Vec2f(pixelPoint[j].x, pixelPoint[j].y);
			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		error = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		error /=  (height * width);   
		totalError += error;
        errorSave.push_back(error);
	}

    return 0;
}


//去畸变 加入了横向畸变和纵向畸变
Point2f undistortmypoints(Point2f xyd , Mat distCoeffs, Mat cameraMatrix)
{
    double x = (xyd.x-cameraMatrix.at<double>(0,2))/cameraMatrix.at<double>(0,0);
    double y = (xyd.y-cameraMatrix.at<double>(1,2))/cameraMatrix.at<double>(1,1);
    double r = sqrt(x*x+y*y);

    // 考虑了切向畸变和径向畸变
    double x_distorted = (1+distCoeffs.at<double>(0,0)*r*r+distCoeffs.at<double>(0,1)*r*r*r*r+distCoeffs.at<double>(0,4)*r*r*r*r*r*r)*x
                                     +2*distCoeffs.at<double>(0,2)*x*y+distCoeffs.at<double>(0,3)*(r*r+2*x*x);
    double y_distorted = (1+distCoeffs.at<double>(0,0)*r*r+distCoeffs.at<double>(0,1)*r*r*r*r+distCoeffs.at<double>(0,4)*r*r*r*r*r*r)*y
                                     +2*distCoeffs.at<double>(0,3)*x*y+distCoeffs.at<double>(0,2)*(r*r+2*y*y);

    double xp = cameraMatrix.at<double>(0,0)*x_distorted+cameraMatrix.at<double>(0,2);
    double yp = cameraMatrix.at<double>(1,1)*y_distorted+cameraMatrix.at<double>(1,2);
    return Point2f(xp,yp);
}

void verificationAccuracy(vector<Mat> &rvecs, vector<Mat> &tvecsleft, vector<vector<Point3f>> &objectPoints, Mat &cameraMatrix,
                        vector<vector<Point2f>> &imagePoint, Mat &distCoeffs
    )
{
    //************************************************对重投影进行测试********************************
    //test the eqution
    /*

      s*x                     X                 X
      s*y = K * [r1 r2 r3 t]* Y = K * [r1 r2 t]*Y
       s                      0                 1
                              1

    */
    //**** 验证外参旋转矩阵R 和 平移向量t的准确性  ****
    // 默认选择 0 1，这里只选择一个点来进行测试，当然，你也可以多试几个
    int k=0,j=1;//k要小于图片的数量，j要小于棋盘纸 高X宽
    Mat Rrw,Rlw;
    cv::Rodrigues(rvecs[k],Rrw);

    Mat po = Mat::zeros(3,3,CV_64F);
    po.at<double>(0,0) = Rrw.at<double>(0,0);
    po.at<double>(1,0) = Rrw.at<double>(1,0);
    po.at<double>(2,0) = Rrw.at<double>(2,0);
    po.at<double>(0,1) = Rrw.at<double>(0,1);
    po.at<double>(1,1) = Rrw.at<double>(1,1);
    po.at<double>(2,1) = Rrw.at<double>(2,1);
    po.at<double>(0,2) = tvecsleft[k].at<double>(0,0);
    po.at<double>(1,2) = tvecsleft[k].at<double>(0,1);
    po.at<double>(2,2) = tvecsleft[k].at<double>(0,2);

    Mat obj(3,1,CV_64F);
    obj.at<double>(0,0) =  objectPoints[k][j].x;
    obj.at<double>(1,0) =  objectPoints[k][j].y;
    obj.at<double>(2,0) =  1;

    Mat uv = cameraMatrix*po*obj;

    Point2f xyd = imagePoint[k][j];
    //对该点进行畸变矫正
    Point2f xyp = undistortmypoints(xyd, distCoeffs, cameraMatrix);

    cout<<"Test the outer parameters（请查看下面两个数据的差距，理论上是一样的）"<<endl;
    cout<<(uv/uv.at<double>(2,0)).t()<<endl; // [x y] = [x/w y/w]
    cout<<xyp<<endl;

    /*
     * 这里需要说明一点，如果上面两个值输入的差距越小，说明重投影的误差小，那么由公式 Rrl = Rrw*Rlw.inv()计算
     * 得到的，即从第一个相机到第二个相机的变换矩阵R的准确率越高；
    */
}

void rotationMatrixVerification(int testObject, vector<Mat> &leftRvecs, vector<Mat> &rightRvecs)
{
     /************************
        Pl = Rlw * Pw+Tlw;
        Pr = Rrw * Pw+Trw;

        --Pr = Rrl * Pl + Trl

       so =====>
                    Rrl = Rrw * Rlw^T
                    Trl = Trw - Rrl * Tlw

    *****************************/
    //**** 验证上述公式是否成立  ****
    //--多试试几组值，如果和最后得到的R差距非常大，很可能标定失败；
    Mat leftRotationMatrix;
    Mat rightRotationMatrix;

    Rodrigues(rightRvecs[testObject],rightRotationMatrix);
    Rodrigues(leftRvecs[testObject],leftRotationMatrix);

    cout << "\n" << "Comparing rightRotationMatrix and R: \n" << endl;
    cout << "rightRotationMatrix = \n" << rightRotationMatrix * leftRotationMatrix.inv() << "\n\n";

    //如果上面Rrw*Rlw.inv() 输出结果 与R输出差距较大，可以去除下列注释测试
    //如果最后标定输出的remap()图像效果很差，也可以去除下列注释看看测试结果
    //R = Rrw*Rlw.inv();                      //Rrl = Rrw * Rlw^T
    //T = tvecsRight[k] - R * tvecsLeft[k];   //Trl = Trw - Rrl * Tlw
}

void outputParam(string str, Mat paramMatrix)
{
    cout << str << "\n" << paramMatrix << "\n\n";
}

int displayCorrectionResults(vector<string> &leftimagename, vector<string> &rightimagename, Mat outmap[TWO][TWO])
{

    for(size_t i = 0; i < leftimagename.size(); ++i)
    {
        Mat leftImageRemap;
        Mat rightImageRemap;

        Mat leftImage = imread(leftimagename[i], 0);
        Mat rightImage = imread(rightimagename[i], 0);

        if(leftImage.empty() || rightImage.empty())
            continue;

        remap(leftImage, leftImageRemap,  outmap[0][0], outmap[0][1], INTER_LINEAR);
        remap(rightImage, rightImageRemap,  outmap[1][0], outmap[1][1], INTER_LINEAR);

        Mat srcAnddst(leftImageRemap.rows,leftImageRemap.cols + rightImageRemap.cols,leftImageRemap.type());

        Mat submat = srcAnddst.colRange(0,leftImageRemap.cols);
        leftImageRemap.copyTo(submat);
        submat = srcAnddst.colRange(leftImageRemap.cols,leftImageRemap.cols + rightImageRemap.cols);
        rightImageRemap.copyTo(submat);

       cvtColor(srcAnddst,srcAnddst,COLOR_GRAY2BGR);

       //draw rectified image
        for (int i = 0; i < srcAnddst.rows;i+=16)
           line(srcAnddst, Point(0, i), Point(srcAnddst.cols, i), Scalar(0, 255, 0), 1, 8);

        imshow("remap",srcAnddst);
        //imshow("ir",irremap);
        waitKey(1000);
    }


    return 0;
}

void saveCorrectionResults(string savepath, Mat leftCameraMatrix, Mat leftDistCoeffs, Mat rightCameraMatrix, Mat rightDistCoeffs,
                           Mat rotationMatrix, Mat translationVector, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q
                )
{
    string yamlFileOutPath = savepath + "stereocalibration.yaml";
	ofstream yamlFile(yamlFileOutPath);
	yamlFile.clear();

    yamlFile << "# ------left camera Intrinsic--------"<<endl;
    yamlFile << "Left.Camera.fx:  " << leftCameraMatrix.at<double>(0,0) << endl;
    yamlFile << "Left.Camera.fy:  " << leftCameraMatrix.at<double>(1,1) << endl;
    yamlFile << "Left.Camera.cx:  " << leftCameraMatrix.at<double>(0,2) << endl;
    yamlFile << "Left.Camera.cy:  " << leftCameraMatrix.at<double>(1,2) << endl << endl;

    yamlFile << "# ------left camera Distortion--------"<<endl;
    yamlFile << "Left.Camera.k1:  " << leftDistCoeffs.at<double>(0,0) << endl;
    yamlFile << "Left.Camera.k2:  " << leftDistCoeffs.at<double>(0,1) << endl;
    yamlFile << "Left.Camera.p1:  " << leftDistCoeffs.at<double>(0,2) << endl;
    yamlFile << "Left.Camera.p2:  " << leftDistCoeffs.at<double>(0,3) << endl;
    yamlFile << "Left.Camera.k3:  " << leftDistCoeffs.at<double>(0,4) << endl<<endl;

    yamlFile << "\n\n\n";

    yamlFile << "# ------right camera Intrinsic--------"<<endl;
    yamlFile << "Right.Camera.fx:  " << rightCameraMatrix.at<double>(0,0) << endl;
    yamlFile << "Right.Camera.fy:  " << rightCameraMatrix.at<double>(1,1) << endl;
    yamlFile << "Right.Camera.cx:  " << rightCameraMatrix.at<double>(0,2) << endl;
    yamlFile << "Right.Camera.cy:  " << rightCameraMatrix.at<double>(1,2) << endl << endl;

    yamlFile << "# ------right camera Distortion--------"<<endl;
    yamlFile << "Right.Camera.k1:  " << rightDistCoeffs.at<double>(0,0) << endl;
    yamlFile << "Right.Camera.k2:  " << rightDistCoeffs.at<double>(0,1) << endl;
    yamlFile << "Right.Camera.p1:  " << rightDistCoeffs.at<double>(0,2) << endl;
    yamlFile << "Right.Camera.p2:  " << rightDistCoeffs.at<double>(0,3) << endl;
    yamlFile << "Right.Camera.k3:  " << rightDistCoeffs.at<double>(0,4) << endl << endl;

    yamlFile.close();

    FileStorage yamlf(yamlFileOutPath, FileStorage::APPEND);
    yamlf.writeComment("------camera parameters saved by yaml data--------", true);
    yamlf << "R"  << rotationMatrix; 
    yamlf << "T"  << translationVector;
    yamlf << "R1" << R1;
    yamlf << "R2" << R2;
    yamlf << "P1" << P1;
    yamlf << "P2" << P2;
    yamlf << "Q"  << Q;

    yamlf.release();

}

int main( int argc, char** argv )
{
    if(argc != 6){
		cerr << "input: ./filename [left image path] [right image path] [boardwidth] [boardheight] [Square side length]" << endl;
		return -1;
	}

    // 输出参数矩阵
	Mat leftCameraMatrix; //输出内参数矩阵
	Mat leftDistCoeffs;   //输出畸变矩阵
	vector<Mat> leftRvecs; //　旋转矩阵
	vector<Mat> leftTvecs; //　平移向量 

    Mat rightCameraMatrix; //输出内参数矩阵
	Mat rightDistCoeffs;   //输出畸变矩阵
	vector<Mat> rightRvecs; //　旋转矩阵
	vector<Mat> rightTvecs; //　平移向量 


    // 左右相机图片的路径
    vector<string> leftImagePath;
    vector<string> rightImagePath;
    glob(argv[1], leftImagePath);
    glob(argv[2], rightImagePath);

    // 存储容器
    vector<vector<Point3f>> leftObjectPoints; 
    vector<vector<Point2f>> leftImagePoint;

    vector<vector<Point3f>> rightObjectPoints; 
    vector<vector<Point2f>> rightImagePoint;

    
    vector<Point2f> leftImageCorner;

    vector<Point2f> rightImageCorner;

    Size imageSize;
    Size newImageSize;

    // 误差存储
    vector<double> leftErrorSaveVector;
    vector<Point2f> leftPixelPoint;
    double leftTotalError;

    vector<double> rightErrorSaveVector;
    vector<Point2f> rightPixelPoint;
    double rightTotalError;

    // 双目标定的参数
    Mat rotationMatrix;     // 旋转矩阵
    Mat translationVector;  // 平移向量
    Mat essentialMatrix;    // 本质矩阵
    Mat foundamentalMatrix; // 基础矩阵

    Mat R1;
    Mat R2;
    Mat P1;
    Mat P2;
    Mat Q;

    Rect validPixROI1;
    Rect validPixROI2;

    Mat outputMap[2][2];
    cout << "\n" << "<------Left Camera intrinsic Matrix------>\n";
    cameraCalibrate(leftImagePath, atoi(argv[4]), atoi(argv[3]), atoi(argv[5]), leftCameraMatrix, leftDistCoeffs, leftImagePoint, 
                    imageSize, leftObjectPoints, leftRvecs, leftTvecs
            );

    cout << "\n" << "<------Right Camera intrinsic Matrix------>\n";
    cameraCalibrate(rightImagePath, atoi(argv[4]), atoi(argv[3]), atoi(argv[5]), rightCameraMatrix, rightDistCoeffs, rightImagePoint, 
                    imageSize, rightObjectPoints, rightRvecs, rightTvecs
            );

    reprojectionError(leftImagePath, atoi(argv[4]), atoi(argv[3]), leftObjectPoints, leftRvecs, leftTvecs, leftCameraMatrix, 
                      leftDistCoeffs, leftPixelPoint, leftImagePoint, leftErrorSaveVector, leftTotalError
    );

    reprojectionError(leftImagePath, atoi(argv[4]), atoi(argv[3]), leftObjectPoints, leftRvecs, leftTvecs, leftCameraMatrix, 
                      leftDistCoeffs, rightPixelPoint, leftImagePoint, rightErrorSaveVector, rightTotalError
    );

    verificationAccuracy(rightRvecs, leftTvecs, leftObjectPoints, leftCameraMatrix, leftImagePoint, leftDistCoeffs);

    cout << "Binocular camera calibration...\n";

    stereoCalibrate( leftObjectPoints, 
                     leftImagePoint,
                     rightImagePoint,
                     leftCameraMatrix,
                     leftDistCoeffs,
                     rightCameraMatrix,
                     rightDistCoeffs,
                     imageSize,
                     rotationMatrix,
                     translationVector,
                     essentialMatrix,
                     foundamentalMatrix,
                     CALIB_FIX_INTRINSIC +
                     CALIB_USE_INTRINSIC_GUESS,
                     TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5)
    );

    rotationMatrixVerification( ZERO,
                                leftRvecs,
                                rightRvecs
    );

    newImageSize = imageSize;

    stereoRectify( leftCameraMatrix,
                   leftDistCoeffs,
                   rightCameraMatrix,
                   rightDistCoeffs,
                   imageSize,
                   rotationMatrix,
                   translationVector,
                   R1,
                   R2,
                   P1,
                   P2,
                   Q,
                   CALIB_ZERO_DISPARITY,
                   NEGATIVE,
                   newImageSize,
                   &validPixROI1,
                   &validPixROI2

    );

    outputParam("rotationMatrix = ", rotationMatrix);
    outputParam("translationVector = ", translationVector);
    outputParam("R1 = ", R1);
    outputParam("R2 = ", R2);
    outputParam("P1 = ", P1);
    outputParam("P2 = ", P2);
    outputParam("Q = ", Q);

    initUndistortRectifyMap( leftCameraMatrix,
                             leftDistCoeffs,
                             R1,
                             P1,
                             imageSize,
                             CV_16SC2,
                             outputMap[0][0],
                             outputMap[0][1]

    );

    initUndistortRectifyMap( rightCameraMatrix,
                             rightDistCoeffs,
                             R2,
                             P2,
                             imageSize,
                             CV_16SC2,
                             outputMap[1][0],
                             outputMap[1][1]

    );

    displayCorrectionResults(leftImagePath, rightImagePath, outputMap);

    cout << "\n<--------save result-------->\n";

    saveCorrectionResults( argv[1],
                           leftCameraMatrix,
                           leftDistCoeffs,
                           rightCameraMatrix,
                           rightDistCoeffs,
                           rotationMatrix,
                           translationVector,
                           R1,
                           R2,
                           P1,
                           P2,
                           Q
    );

    return 0;
}






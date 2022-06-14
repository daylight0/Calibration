#include <iostream>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv4/opencv2/highgui.hpp>
 
using namespace std;
using namespace cv;

 
int main(int argc, char** argv)
{
	if(argc != 6){
		cerr << "input: ./filename [imageWidth] [imageHeight] [leftimage path] [rightimage path] [camera]" << endl;
		cerr << "example: ./stereo 1280 480 ../data/left/ ../data/right/ 0" << endl;
		return -1;
	}

	Mat image, leftImage, rightImage;
    int index = 0;

	VideoCapture camera(atoi(argv[5]), CAP_V4L2); //后面的参数必须设置为CAP_V4L2或者CAP_V4L,否则双目输出有问题

	if (!camera.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
	//设置摄像头的采集参数
	camera.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	camera.set(CAP_PROP_FRAME_WIDTH, atoi(argv[1]));
	camera.set(CAP_PROP_FRAME_HEIGHT, atoi(argv[2]));
	camera.set(CAP_PROP_FPS, 30);

	string leftImagePath(argv[3]);
	string rightImagePath(argv[4]);

	camera >> image;
	if (image.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return -1;
    }

	// 成功打开进入循环
	while(1){
		if (!camera.read(image)) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
		
        leftImage = image(Rect(0, 0, atoi(argv[1]) / 2, atoi(argv[2])));
        rightImage = image(Rect(atoi(argv[1]) / 2, 0, atoi(argv[1]) / 2, atoi(argv[2])));
		imshow("left image", leftImage);
		imshow("right image", rightImage);
		int key = waitKey(30);
		if(key == 'q') break;
		if(key == 'c'){
			++index;
			imwrite(leftImagePath+"/"+to_string(index)+".jpg",leftImage);
			imwrite(rightImagePath+"/"+to_string(index)+".jpg",rightImage);
		}
	}

	camera.release();
	destroyAllWindows();

	return 0;
}
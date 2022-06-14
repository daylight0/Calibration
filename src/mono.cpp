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
	if(argc != 5){
		cerr << "input: ./filename [imageWidth] [imageHeight] [image path] [camera interface]" << endl;
		cerr << "example: ./stereo 1280 480 ../data/image/ 0" << endl;
		return -1;
	}

	Mat image;
    int index = 0;

	VideoCapture camera(atoi(argv[4]), CAP_V4L2); //后面的参数必须设置为CAP_V4L2或者CAP_V4L,否则双目输出有问题

	if (!camera.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
	//设置摄像头的采集参数
	camera.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
	camera.set(CAP_PROP_FRAME_WIDTH, atoi(argv[1]));
	camera.set(CAP_PROP_FRAME_HEIGHT, atoi(argv[2]));
	camera.set(CAP_PROP_FPS, 30);

	string ImagePath(argv[3]);

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
		
		imshow("image", image);
		int key = waitKey(30);
		if(key == 'q') break;
		if(key == 'c'){
			++index;
			imwrite(ImagePath+"/"+to_string(index)+".jpg",image);
		}
	}

	camera.release();
	destroyAllWindows();

	return 0;
}
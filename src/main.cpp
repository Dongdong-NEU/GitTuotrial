#include "orb_extractor.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace XIAOC;

int main( int argc, char** argv )
{
    cout << "Enter the ORBextractor module..." << endl;
    
    // help infomation
    if ( argc != 2 )
    {
        cout << "Please input ./orb_extractor image" << endl;
        return -1;
    }

    // -----grid based orb extractor
    int nfeatures = 1000;
    int nlevels = 8;
    float fscaleFactor = 1.2;
    float fIniThFAST = 20;
    float fMinThFAST = 7;
    // default parameters printf
    cout << "Default parameters are : " << endl;
    cout << "nfeature : " << nfeatures << ", nlevels : " << nlevels << ", fscaleFactor : " << fscaleFactor << endl;
    cout << "fIniThFAST : " << fIniThFAST << ", fMinThFAST : " << fMinThFAST << endl;

    // read image
    cout << "Read image..." << endl;
    Mat image = imread( argv[1], CV_LOAD_IMAGE_UNCHANGED );

    Mat grayImg, mask;
    cvtColor( image, grayImg, CV_RGB2GRAY );
    imshow( "grayImg", grayImg );
    waitKey( 0 );
    cout << "Read image finish!" << endl;

    // orb extractor initialize
    cout << "ORBextractor initialize..." << endl;
    ORBextractor* pORBextractor;
    pORBextractor = new ORBextractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
    cout << "ORBextractor initialize finished!" << endl;


    // orb extractor
    cout << "Extract orb descriptors..." << endl;
    Mat desc;
    vector<KeyPoint> kps;
    (*pORBextractor)( grayImg, mask, kps, desc );
    cout << "Extract orb descriptors finished!" << endl;

    cout << "The number of keypoints are = " << kps.size() << endl;

    // draw keypoints in output image
    Mat outImg;
    drawKeypoints( grayImg, kps, outImg, cv::Scalar::all(-1), 0 );
    imshow( "GridOrbKpsImg", outImg );
    waitKey( 0 );
    cout << "Finished! Congratulations!" << endl;

    /**
     * 该段程序绘制两段两帧图像中成功追踪的光流点;
     */

    Mat Tracked_OpticalFlow;
    cv::cvtColor(grayImg, Tracked_OpticalFlow, CV_GRAY2BGR);
    for (int i = 0; i < kps.size(); i++) {
        Point2f orbdot = kps[i].pt;
        cv::circle(Tracked_OpticalFlow, orbdot, 1, cv::Scalar(225, 105, 65), 2);//半径,
    }
    imshow( "Tracked_OpticalFlow", Tracked_OpticalFlow );
    waitKey( 0 );

    string path_save_Tracked_OpticalFlow="/media/xhd/ShareDate/Linux/DynamicDatebase/框图图片数据/3.w-static/优选/Tracked_OpticalFlow3.png";  //将当前语义分割图保存到硬盘里；
    cv::imwrite(path_save_Tracked_OpticalFlow,Tracked_OpticalFlow);

    cv::imshow("Tracked_OpticalFlow", Tracked_OpticalFlow);
    cv::waitKey(0);


    //// ----original orb extractor for comparation
    // orb initialization 
    cout << "Using original orb extractor to extract orb descriptors for comparation." << endl;
    Ptr<ORB> orb_ = ORB::create( 1000, 1.2f, 8, 19 );

    // orb extract
    vector<KeyPoint> orb_kps;
    Mat orb_desc;
    orb_->detectAndCompute( grayImg, mask, orb_kps, orb_desc );

    // draw keypoints in output image
    Mat orbOutImg;
    drawKeypoints( grayImg, orb_kps, orbOutImg, cv::Scalar::all(-1), 0 );
    imshow( "OrbKpsImg", orbOutImg );
    waitKey(0);

    // destroy all windows when you press any key
    //destroyAllWindows();

    return 0;
}
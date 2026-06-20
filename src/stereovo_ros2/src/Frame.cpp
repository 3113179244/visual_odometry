#include "Frame.h"

Frame::Frame(const cv::Mat& left, const cv::Mat& right, double timestamp) 
    : mImgLeft(left), mImgRight(right), mTimeStamp(timestamp) {
    // 构造函数仅存储图像和时间戳
}
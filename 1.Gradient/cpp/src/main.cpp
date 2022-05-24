#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"

enum Direction{ X, Y};

float get_pixel_value(cv::Mat input, int row, int col){
    if ((row < 0) || (col < 0) || (row >= input.rows) || (col >= input.cols)){
        return 0;
    }
    else{
        return input.at<uchar>(row, col);
    }
}

cv::Mat gradient(cv::Mat image, Direction direction){
    cv::Mat output = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
    float prev=0, next=0;
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            // std::cout << i << " " << j << std::endl;
            if (direction == X){
                prev = get_pixel_value(image, i, j-1);
                next = get_pixel_value(image, i, j+1);
            }
            else{
                prev = get_pixel_value(image, i-1, j);
                next = get_pixel_value(image, i+1, j);
            }
            output.at<uchar>(i,j) = next + prev - 2*(get_pixel_value(image, i, j));
        }
    }
    return output;
}

cv::Mat zero_crossing(cv::Mat image, Direction direction){
    cv::Mat output = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
    float prev=0, next=0;
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            if (direction == X){
                prev = get_pixel_value(image, i, j-1);
                next = get_pixel_value(image, i, j+1);
            }
            else{
                prev = get_pixel_value(image, i-1, j);
                next = get_pixel_value(image, i+1, j);
            }
            if (get_pixel_value(image, i, j) == 0){
                output.at<uchar>(i,j) = 255;
            }
            else if ((prev > 0 && next <=0) || (prev <0 && next >=0)){
                output.at<uchar>(i,j) = 255;
            }
        }
    }
    return output;
}

int main(){
    std::string INPUT_IMAGE{"images/tools.png"};
    cv::Mat image = cv::imread(INPUT_IMAGE, cv::IMREAD_GRAYSCALE);
    cv::Mat grad_x{gradient(image, X)};
    cv::Mat grad_y{gradient(image, Y)};
    cv::Mat grad{grad_x.mul(grad_x) + grad_y.mul(grad_y)};
    

    cv::Mat zero_crossing_x{zero_crossing(grad, X)};
    cv::Mat zero_crossing_y{zero_crossing(grad, Y)};
    cv::Mat zero_crossing{zero_crossing_x | zero_crossing_y};
    cv::imshow("original",image);
    cv::imshow("grad", grad);
    cv::imshow("zero crossing", zero_crossing);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
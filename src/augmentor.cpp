/*********************************************************************
Author : Hyunki Seong
Date   : 2019.08.24

Image Augmentor

*********************************************************************/

#include <iostream>
#include <stdio.h>
#include <fstream> // for loading txt file
#include <sstream> // for parsing data in txt using std::stringstream
#include <opencv2/highgui.hpp>
#include <omp.h> // for OpemMP, parallel processing
#include <chrono>

/* ========== Parameters ========== */
#define IMSHOW_ON false
#define IMAGE_RAW_PATH  "/home/user/Documents/extracted_data/label.txt"
#define IMAGE_SAVE_PATH "/home/user/Documents/augment_data"
#define STEER_SAVE_PATH "/home/user/Documents/augment_data"
#define ROI_SHIFT_CNT  4
#define ROI_SHIFT_RES  35
#define ROI_LEFTTOP_X  160
#define ROI_LEFTTOP_Y  0 // 374 w.r.t raw_image, 0 w.r.t. crop_image
#define ROI_WIDTH      960
#define ROI_HEIGHT     650
#define STEER_PER_ROI_SHIFT 12
#define COLOR_AUG_CNT  4
#define COLOR_AUG_RES  4

using namespace std::chrono;
///https://stackoverflow.com/a/19555298/9609025
long curTime()
{
    milliseconds ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch());

    return ms.count();
}

int main(int argc, char **argv)
{
    clock_t start, end; // for checking process time

    /* Load txt file */
    std::string in_line_for_check;
    std::ifstream in_for_check(IMAGE_RAW_PATH);
    int line_iter = 0;
    int line_total = 0;
    while(getline(in_for_check, in_line_for_check)) // for counting total line
    {
        line_total++;
    }

    std::string in_line;
    std::ifstream in(IMAGE_RAW_PATH);

    std::string image_save_path = IMAGE_SAVE_PATH;
    std::string steer_save_path = STEER_SAVE_PATH;

    /* ========== Read each line in txt ========== */
    while(getline(in, in_line))
    {
        std::cout << "Current progress : " << line_iter << " / " << line_total << std::endl;

        std::stringstream ss(in_line);
        std::string path;       // data type for path
        int steer;              // data type for steer
        std::string timestamp;  // data type for timestamp
        
        ss >> path;       // parsing path from ss to path
        ss >> steer;      // parsing steer from ss to steer
        ss >> timestamp;  // parsing timestamp from ss to timestamp

        std::cout << "Start process for [path : " << path << " | steer : " << steer << " | timestamp : " << timestamp << "]" << std::endl;

        /* ========== Load image ========== */
        cv::Mat image_raw = cv::imread(path, CV_LOAD_IMAGE_UNCHANGED);
        // Show image
        if (IMSHOW_ON)
        {
            cv::imshow("image raw ", image_raw);
            cv::waitKey(0);
            cv::destroyWindow("image raw");
        }

        /* ========== Image Augmentation and Steer preprocessing ========== */
        start = curTime();

        /* =====================
        Set Parallel Processing
        int idx;
        omp_set_num_threads(8);
        #pragma omp parallel for
        for (idx = ...)
        =======================*/

        int shift_cnt;
        omp_set_num_threads(16);
        #pragma omp parallel for

        // AUGMENTATION BY ROI SHIFTING
        for(shift_cnt = -1*ROI_SHIFT_CNT; shift_cnt <= ROI_SHIFT_CNT; shift_cnt++)
        {
            cv::Rect rect(ROI_LEFTTOP_X + shift_cnt*ROI_SHIFT_RES, ROI_LEFTTOP_Y, ROI_WIDTH, ROI_HEIGHT); // raw image: 1280x1024
            cv::Mat image_crop = image_raw(rect);       // crop image
            cv::Mat color_auged_image = cv::Mat::zeros( image_crop.size(), image_crop.type() );
            for(int color_aug_cnt = -1*COLOR_AUG_CNT; color_aug_cnt <= COLOR_AUG_CNT; color_aug_cnt++)
            {
                // AUGMENTATION BY COLOR CHANGE
                for(int y = 0; y < image_crop.rows; y++ )
                {
                    for(int x = 0; x < image_crop.cols; x++ )
                    {
                        for(int c = 0; c < image_crop.channels(); c++ )
                        {
                            color_auged_image.at<cv::Vec3b>(y,x)[c] =
                            cv::saturate_cast<uchar>( image_crop.at<cv::Vec3b>(y,x)[c] + (color_aug_cnt*COLOR_AUG_RES));
                        }
                    }
                }

                /* ========== Save Image and Steer data ========== */
                // SAVE AUGMENTED IMAGE
                std::string filename = image_save_path + "/" + timestamp + "_R_" + std::to_string(shift_cnt*ROI_SHIFT_RES)+ "_C_" + 
                    std::to_string(color_aug_cnt * COLOR_AUG_RES) + "_" + std::to_string(steer + shift_cnt*STEER_PER_ROI_SHIFT) + ".jpg"; // path for crop image
                cv::imwrite(filename, color_auged_image);

                // Save Steer data in the txt file.
                std::ofstream label_steer(steer_save_path + "/label.txt", std::ios::app);
                label_steer << filename << " " << std::to_string(steer + shift_cnt*STEER_PER_ROI_SHIFT) << std::endl; // Save Steering curvature value
            }
        }
        end = curTime();
        unsigned long diff1=end-start;
        // std::cout << "Process time : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Process time [s] : " << (float)(diff1/1000.0) << std::endl;
        
        line_iter++; // for checking progress
    }
    
    return 0;
    }


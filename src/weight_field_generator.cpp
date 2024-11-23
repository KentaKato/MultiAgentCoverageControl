#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main()
{
    std::string image_dir = "/home/kenta-kato/ws/MultiAgentCoverageControl/image/";
    std::string image_file = "star.png";
    std::string image_path = image_dir + image_file;
    cv::Mat image = cv::imread(image_dir + image_file, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return EXIT_FAILURE;
    }

    // Create a binary image (black is 1, white is 0)
    cv::Mat binary_image;
    cv::threshold(image, binary_image, 127, 1, cv::THRESH_BINARY_INV);

    // Distance transform (distance from black pixels)
    cv::Mat distance;
    cv::distanceTransform(1 - binary_image, distance, cv::DIST_L2, 3);

    // Get the maximum distance
    double max_dist;
    cv::minMaxLoc(distance, nullptr, &max_dist);

    constexpr double MAX = 254.0;
    constexpr double MIN = 1.0;
    // Calculate weights (100 at distance 0, 0 at distance max_dist or more)
    cv::Mat weights = MAX *
        (1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist)
        .mul(1.0 - distance / max_dist);

    // Limit the value to the range 0 to MAX
    cv::threshold(weights, weights, 0, 0, cv::THRESH_TOZERO);
    cv::threshold(weights, weights, MAX, MAX, cv::THRESH_TRUNC);

    // clip the value to the range MIN to MAX
    weights.setTo(MIN, weights < MIN);
    weights.setTo(MAX, binary_image);

    // min on weights
    double min_val, max_val;
    cv::minMaxLoc(weights, &min_val, &max_val);
    std::cout << "min: " << min_val << ", max: " << max_val << std::endl;


    // Convert to 8-bit image for visualization
    cv::Mat display_image;
    weights.convertTo(display_image, CV_8UC1);

    // Apply a color map
    // cv::applyColorMap(display_image, display_image, cv::COLORMAP_HOT);

    cv::imshow("Input Image", image);
    cv::imshow("Weighted Field", display_image);
    cv::waitKey(0);

    std::string output_file = "weighted_field.png";
    std::string output_path = image_dir + output_file;
    cv::imwrite(output_path, display_image);

    return EXIT_SUCCESS;
}
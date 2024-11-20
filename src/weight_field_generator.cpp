#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::string imagePath = "/home/kenta-kato/ws/DelaunayVoronoiCpp/image/star.png";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return EXIT_FAILURE;
    }

    // Create a binary image (black is 1, white is 0)
    cv::Mat binaryImage;
    cv::threshold(image, binaryImage, 127, 1, cv::THRESH_BINARY_INV);

    // Distance transform (distance from black pixels)
    cv::Mat distance;
    cv::distanceTransform(1 - binaryImage, distance, cv::DIST_L2, 3);

    // Get the maximum distance
    double maxDist;
    cv::minMaxLoc(distance, nullptr, &maxDist);

    // Calculate weights (100 at distance 0, 0 at distance maxDist or more)
    cv::Mat weights = 100.0 * (1.0 - distance / maxDist);

    // Limit the value to the range 0 to 100
    cv::threshold(weights, weights, 0, 0, cv::THRESH_TOZERO);
    cv::threshold(weights, weights, 100, 100, cv::THRESH_TRUNC);

    // Set the maximum value of 100 to black pixels
    weights.setTo(100, binaryImage);

    // Convert to 8-bit image for visualization
    cv::Mat displayImage;
    weights.convertTo(displayImage, CV_8UC1);

    // Apply a color map
    cv::applyColorMap(displayImage, displayImage, cv::COLORMAP_HOT);

    cv::imshow("Input Image", image);
    cv::imshow("Weighted Field", displayImage);
    cv::waitKey(0);

    cv::imwrite("weighted_field.png", displayImage);

    return EXIT_SUCCESS;
}
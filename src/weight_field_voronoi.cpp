#include "DelaunayTriangulation/delaunay_triangulation.hpp"
#include "DelaunayTriangulation/delaunay_triangulation_drawer.hpp"
#include "DelaunayTriangulation/voronoi_diagram.hpp"

#include <limits>
#include <random>
#include <algorithm>

using namespace delaunay_triangulation;

void addRandomVertices(DelaunayTriangulation &delaunay, int image_width, int image_height, size_t num_vertices)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> width_dist(0, image_width);
    std::uniform_real_distribution<> height_dist(0, image_height);
    for (size_t i = 0; i < num_vertices; ++i)
    {
        Vertex v(width_dist(engine), height_dist(engine));
        if (!delaunay.hasVertex(v))
        {
            delaunay.addVertex(v);
        }
        else {
            --i;
        }
    }
}


int main()
{
    // Initial image
    std::string image_dir = "/home/kenta-kato/ws/DelaunayVoronoiCpp/image/";

    /* image 1 */
    std::string image_file = "1.png";
    std::string image_path = image_dir + image_file;
    cv::Mat weight_field_img_1 = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (weight_field_img_1.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return EXIT_FAILURE;
    }
    int image_width = weight_field_img_1.cols;
    int image_height = weight_field_img_1.rows;
    std::cout << "image_width: " << image_width << std::endl;
    std::cout << "image_height: " << image_height << std::endl;

    //Store the value of each pixel in weight_field_image in weight_map
    std::map<Point, double> weight_map_1;
    int step = 2;
    double max_weight = std::numeric_limits<double>::min();
    double min_weight = std::numeric_limits<double>::max();
    for (int x = 0; x < weight_field_img_1.cols; x += step)
    {
        for (int y = 0; y < weight_field_img_1.rows; y += step)
        {
            const uchar pixel_value = weight_field_img_1.at<uchar>(y, x);
            const double w = static_cast<double>(pixel_value);
            if (w > max_weight)
            {
                max_weight = w;
            }
            if (w < min_weight)
            {
                min_weight = w;
            }
            weight_map_1[Point{static_cast<double>(x), static_cast<double>(y)}] = w;
        }
    }


    /* image 1 */
    image_file = "2.png";
    image_path = image_dir + image_file;
    cv::Mat weight_field_img_2 = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (weight_field_img_2.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return EXIT_FAILURE;
    }


    //Store the value of each pixel in weight_field_image in weight_map
    std::map<Point, double> weight_map_2;
    for (int x = 0; x < weight_field_img_2.cols; x += step)
    {
        for (int y = 0; y < weight_field_img_2.rows; y += step)
        {
            const uchar pixel_value = weight_field_img_2.at<uchar>(y, x);
            const double w = static_cast<double>(pixel_value);
            weight_map_2[Point{static_cast<double>(x), static_cast<double>(y)}] = w;
        }
    }

    // Initialize display image
    cv::Scalar white(255, 255, 255);
    cv::Mat img = cv::Mat(image_height, image_width, CV_8UC3, white);


    DelaunayTriangulation delaunay;
    DelaunayTriangulationDrawer drawer(delaunay);
    drawer.setFillTriangle(false);

    size_t num_vertices = 80;
    addRandomVertices(delaunay, image_width, image_height, num_vertices);


    // initialize belonging cells
    std::map<Point, Site> belonging_cells;
    for (const int &x : {0, image_width})
    {
        for (const int &y : {0, image_height})
        {
            Point p{static_cast<double>(x), static_cast<double>(y)};
            Site s;
            belonging_cells[p] = s;
        }
    }


    size_t count = 0;
    auto & weight_map = weight_map_1;
    while (true)
    {
        count++;
        if (count > 50)
        {
            weight_map = weight_map_2;
        }

        // Draw the weight map
        img.setTo(white);
        // for (const auto & [p, w]: weight_map)
        // {
        //     cv::circle(img, cv::Point(p.x, p.y), 1, cv::Scalar(100, 0, static_cast<int>((w - min_weight) / (max_weight - min_weight) * 100)), -1, cv::LINE_AA);
        // }

        delaunay.createDelaunayTriangles();
        auto voronoi_cells = VoronoiDiagram::create(delaunay.getAllTriangles());
        // VoronoiDiagram::draw( img, voronoi_cells);

        std::vector<Site> sites;
        sites.reserve(voronoi_cells.size());
        for (const auto & [site, _] : voronoi_cells)
        {
            sites.push_back(site);
        }

        for (const auto & [point, _]: belonging_cells)
        {
            VoronoiDiagram::findBelongingCell(sites, point, belonging_cells[point]);
        }

        std::map<Site, Centroid> voronoi_centroids;
        VoronoiDiagram::computeVoronoiCentroids(
            sites,
            weight_map,
            voronoi_centroids);

        cv::Scalar centroid_color(0, 255, 0);
        for (const auto & [site, centroid] : voronoi_centroids)
        {
            // centroid.draw(img, false, centroid_color);
            site.draw(img, false);
        }

        cv::imshow("Centroid Voronoi Diagram", img);

        cv::waitKey(1);

        delaunay.clear();
        delaunay.reserveVerticesVector(num_vertices);


        for (const auto & [site, centroid] : voronoi_centroids)
        {
            Site next_site;;
            double move_x = centroid.x - site.x;
            double move_y = centroid.y - site.y;
            constexpr double max_move = 7.0;
            next_site.x = site.x + std::clamp(move_x, -max_move, max_move);
            next_site.y = site.y + std::clamp(move_y, -max_move, max_move);

            delaunay.addVertex(next_site);
        }
    }
}


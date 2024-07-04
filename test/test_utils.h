#pragma once
#include <experimental/filesystem>
#include <set>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#define PRINT_GREEN(x) std::cout << "\033[1;32m" << x << "\033[0m" << std::endl;
#define PRINT_RED(x) std::cout << "\033[1;31m" << x << "\033[0m" << std::endl;
#define PRINT_YELLOW(x) std::cout << "\033[1;33m" << x << "\033[0m" << std::endl;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

namespace fs = std::experimental::filesystem;

std::vector<std::string> loadImagePaths(const std::string &directory) {
  std::vector<std::string> images;
  for (auto &p : fs::directory_iterator(directory)) {
    if (p.path().extension() == ".jpg") {
      images.emplace_back(p.path().string());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

int load_loop_closures(const std::string &path2file, std::vector<std::pair<int, int>> &data)
{
    std::ifstream infile(path2file);
    if (!infile) {
        std::cerr << "Unable to open file";
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int val1, val2, val4, val5;
        double val3;

        if (!(iss >> val1 >> val2 >> val3 >> val4 >> val5)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }

        data.push_back(std::make_pair(val1, val2));
    }

    infile.close();
    return 0;
}

int load_scores(const std::string &path2file, 
	std::vector<std::pair<int, float>> &data) 
{
    
	std::ifstream file(path2file);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return 1;
    }

    int key;
    float value;
    char delimiter;

    while (file >> key >> delimiter >> value) {
        if (delimiter == ',') {
            data.push_back(std::make_pair(key, value));
        }
    }

    file.close();
    return 0;
}

bool has_extension(const std::string &file, const std::vector<std::string> &exts)
{
    for (const auto &ext : exts)
    {
        if (file.length() >= ext.length())
        {
            if (0 == file.compare(file.length() - ext.length(), ext.length(), ext))
            {
                return true;
            }
        }
        else
        {
            continue;
        }
    }
    return false;
}

void get_files(std::string dir_name, std::vector<std::string> &files_in_dir, std::vector<std::string> extension = {".png", ".jpg", ".tif", ".bmp"})
{
    DIR *dir;
    struct dirent *ent;
    // std::string path = "/home/gvasserm/Downloads/Bicocca_Static_Lamps/temp/"; // Change this to your directory path

    if ((dir = opendir(dir_name.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string file_name = ent->d_name;
            if (has_extension(file_name, extension))
            {
                // std::cout << file_name << std::endl;
                files_in_dir.push_back(dir_name + file_name);
            }
        }
        closedir(dir);
    }
}

void loadDetectCompute(std::string fname,
                       std::vector<cv::KeyPoint> &keypoints,
                       cv::Mat &features)
{
    cv::Mat im = cv::imread(fname);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    orb->detectAndCompute(im, cv::Mat(), keypoints, features);
}

cv::Mat load_descriptors(const std::string &file_path)
{
    // Create a FileStorage object for reading
    cv::FileStorage file_storage(file_path, cv::FileStorage::READ);

    // Read the descriptors
    cv::Mat descriptors;
    file_storage["desc"] >> descriptors;

    // Release the file
    file_storage.release();

    return descriptors;
}

void printProgress(double percentage)
{
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

std::vector<std::pair<int, double>> read_scores(const std::string &file_path)
{
    std::vector<std::pair<int, double>> data;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open the file " << file_path << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string item;
        int key;
        double value;

        // Read the first value (int)
        std::getline(ss, item, ',');
        key = std::stoi(item);

        // Read the second value (double)
        std::getline(ss, item, ',');
        value = std::stod(item);

        data.emplace_back(key, value);
    }

    file.close();
    return data;
}

// Function to extract the integer ID from a given path
int extract_id(const std::string &path)
{
    // Find the position of the last '/' character
    size_t pos = path.find_last_of("/\\");
    std::string filename;

    if (pos != std::string::npos)
    {
        // Extract the filename
        filename = path.substr(pos + 1);
    }
    else
    {
        // If '/' is not found, the path itself is the filename
        filename = path;
    }

    // Find the position of the first digit in the filename
    size_t digit_pos = filename.find_first_of("0123456789");

    if (digit_pos != std::string::npos)
    {
        // Extract the integer part from the filename
        std::string id_str;
        while (digit_pos < filename.length() && std::isdigit(filename[digit_pos]))
        {
            id_str += filename[digit_pos];
            ++digit_pos;
        }
        return std::stoi(id_str); // Convert the extracted string to an integer
    }

    // Return -1 if no digit is found in the filename
    return -1;
}

void write_scores_to_csv(std::vector<float> &scores, std::map<int,int> &id_map, const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << file_path << std::endl;
        return;
    }
    // Write the header
    file << "id,score\n";

    // Write each Result to the file
    for (size_t i = 0; i < scores.size(); ++i) {
        int id = id_map[i];
        file <<  id << "," << scores[i] << "\n";
    }

    file.close();
}
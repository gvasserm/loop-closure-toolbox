#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <dirent.h>

#include "DBoW2.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace DBoW2;
using namespace std;

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

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

void trainVocDetectCompute()
{
    srand(0);
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L1_NORM;

    OrbVocabulary voc(k, L, weight, scoring);

    std::vector<std::string> test_names;

    std::string test_dir = "/home/gvasserm/dev/aicv_amr_ws/results_orb_old_default_ptk/";

    get_files(test_dir, test_names);
    std::sort(test_names.begin(), test_names.end());
    PRINT_YELLOW("[DBoW3::Database::add] start");

    std::vector<std::vector<cv::Mat>> features;
    for (size_t i = 0; i < test_names.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat feature;
        loadDetectCompute(test_names[i], keypoints, feature);
        //cv::Mat featureF;feature.convertTo(featureF, CV_32F);
        features.push_back(toDescriptorVector(feature));
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    voc.create(features);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << " msecs" << std::endl;
    
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("test_orb_10_6.yml.gz");
    cout << "Done" << endl;
}

void trainVocPrecomputed()
{
    srand(0);
    const int k = 10;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L1_NORM;

    OrbVocabulary voc(k, L, weight, scoring);

    std::string test_dir = "/home/gvasserm/dev/aicv_amr_ws/results_orb_old_default_ptk/";
    std::vector<std::string> extension={".yml"};
    std::vector<std::string> test_names;
    get_files(test_dir, test_names, extension);
    PRINT_YELLOW("[FBOW::Database::add] start");

    std::vector<std::vector<cv::Mat>> features(test_names.size());
    for(size_t i=0;i<test_names.size(); ++i)
    {
        //printProgress(i / 100.0);
        cv::Mat feature = load_descriptors(test_names[i]);
        //cv::Mat featureF;feature.convertTo(featureF, CV_32F);
        features[i]=toDescriptorVector(feature);
    }
    PRINT_YELLOW("[FBOW::Database::add] end");

    auto t_start = std::chrono::high_resolution_clock::now();
    voc.create(features);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << " msecs" << std::endl;
    
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("test_orb_10_6.yml.gz");
    cout << "Done" << endl;
}

void write_results_to_csv(const QueryResults& results, std::map<int,int> &id_map, const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << file_path << std::endl;
        return;
    }

    // Write the header
    file << "Id,Score\n";

    // Write each Result to the file

    for (const auto& result : results) {
        int id = id_map[int(result.Id)];
        file <<  id << "," << result.Score << "\n";
    }

    file.close();
}

void testDBowVoc()
{
    PRINT_YELLOW("[Loading vocabulary] start");
    std::cout << "Current path is " << fs::current_path() << '\n';
    OrbVocabulary voc("test_orb_10_6.yml.gz");
    OrbDatabase db(voc, false, 0); // false = do not use direct index
    PRINT_GREEN("[Loading vocabulary] end\n");

    int frameID = 224;

    std::vector<std::string> test_names;
    std::string test_dir = "/home/gvasserm/dev/aicv_amr_ws/results_orb_old_default_ptk/";
    std::vector<std::pair<int, double>> data = read_scores(test_dir + std::to_string(frameID) + ".csv");

    std::vector<std::string> extension = {".yml"};
    get_files(test_dir, test_names, extension);
    std::sort(test_names.begin(), test_names.end());
    PRINT_YELLOW("[FBOW::Database::add] start");

    std::map<int, std::string> fnames_map;
    std::map<int, int> id_map;
    for (size_t i = 0; i < test_names.size(); ++i)
    {
        int id = extract_id(test_names[i]);
        fnames_map[id] = test_names[i];
    }

    //std::vector<cv::Mat> features;
    for (size_t i = 0; i < data.size(); ++i)
    {
        printProgress(i / 100.0);
        int ind = data[i].first;
        if (ind < 0){
            continue;
        }
        cv::Mat feature = load_descriptors(fnames_map[ind]);
        //cv::Mat featureF;feature.convertTo(featureF, CV_32F);
        db.add(toDescriptorVector(feature));
        //features.push_back(feature);
        id_map[i] = ind;
    }
    PRINT_GREEN("[FBOW::Database::add] end\n");

    PRINT_YELLOW("[FBOW::Database::query] start");
    cv::Mat feature = load_descriptors(fnames_map[frameID]);
    //cv::Mat featureF;feature.convertTo(featureF, CV_32F);
    QueryResults ret;
    db.query(toDescriptorVector(feature), ret, -1);
    write_results_to_csv(ret,  id_map, "results.csv");
    PRINT_GREEN("[FBOW::Database::query] end");
}

int main()
{
    testDBowVoc();
    //trainVocPrecomputed();
    return 0;
}
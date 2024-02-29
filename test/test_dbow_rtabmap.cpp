#include "DBoW3.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include <dirent.h>

bool has_extension(const std::string& file, const std::vector<std::string>& exts) {
    for (const auto &ext : exts) 
	{
		if (file.length() >= ext.length()) {
			if (0 == file.compare(file.length() - ext.length(), ext.length(), ext)){
				return true;
			}
		} 
		else {
			continue;
		}
	}
	return false;
}

void get_files(std::string dir_name, std::vector<std::string> &files_in_dir)
{
	DIR *dir;
    struct dirent *ent;
    //std::string path = "/home/gvasserm/Downloads/Bicocca_Static_Lamps/temp/"; // Change this to your directory path
    std::vector<std::string> extension = {".png", ".jpg", ".tif", ".bmp"}; // Change this to the desired extension

	if ((dir = opendir(dir_name.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file_name = ent->d_name;
            if (has_extension(file_name, extension)) {
                //std::cout << file_name << std::endl;
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

void testDBowDatabase() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  DBoW3::Vocabulary voc(10, 5);
  voc.load("./config/orbvoc.dbow3");
  DBoW3::Database db(voc, false, 0); // false: do not use direct index (default)
  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");

  std::vector<std::string> test_names;
  std::vector<std::string> query_names;

  std::string test_dir = "/home/gvasserm/dev/rtabmap/data/samples/";
    
  get_files(test_dir, test_names);
	std::sort(test_names.begin(), test_names.end());
  PRINT_YELLOW("[DBoW3::Database::add] start");

  std::vector<cv::Mat> features;
  std::vector<DBoW3::BowVector> bvectors;
  for (size_t i=0;i<test_names.size(); ++i)
  {
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat feature;
      loadDetectCompute(test_names[i], keypoints, feature);
      DBoW3::BowVector bv;
      voc.transform(feature, bv);
      db.add(feature);
      //db.add(bv);
      features.push_back(feature);
      bvectors.push_back(bv);
  }
  std::cout << db << std::endl;
  PRINT_GREEN("[DBoW3::Database::add] end\n");


  PRINT_YELLOW("[DBoW3::Database::query] start");

  DBoW3::QueryResults results;
  db.query(features[2], results, -1);
  //db.query(bvectors[2], results, -1);
  std::cout << "Query results: " << results << std::endl;
  PRINT_GREEN("[DBoW3::Database::query] end");
}


int main() {
  testDBowDatabase();
  return 0;
}
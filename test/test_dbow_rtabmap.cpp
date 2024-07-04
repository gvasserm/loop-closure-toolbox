#include "DBoW3.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include "test_utils.h"


void trainVocORB()
{
    DBoW3::Vocabulary voc(10, 5);

    std::vector<std::string> test_names;
    std::vector<std::string> query_names;

    std::string test_dir = "/home/gvasserm/dev/rtabmap/data/samples/";
    
    get_files(test_dir, test_names);
    std::sort(test_names.begin(), test_names.end());
    PRINT_YELLOW("[DBoW3::Database::add] start");

    std::vector<cv::Mat> features;
    for (size_t i=0;i<test_names.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat feature;
        loadDetectCompute(test_names[i], keypoints, feature);
        cv::Mat featureF;
        feature.convertTo(featureF, CV_32F);
        features.push_back(featureF);
    }

    voc.create(features);
    voc.save("config/test_orb_10_5.yaml");
}

void trainVocDesc()
{
    DBoW3::Vocabulary voc(10, 5);

    std::vector<std::string> test_names;
    std::vector<std::string> query_names;
    
    std::string test_dir = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/";

    std::vector<std::string> extension={".yml"};
    get_files(test_dir, test_names, extension);
    std::sort(test_names.begin(), test_names.end());
    PRINT_YELLOW("[DBoW3::Database::add] start");

    std::vector<cv::Mat> features;
    //for (size_t i=0;i<test_names.size(); ++i)
    for (size_t i=0;i<50; ++i)
    {
        printProgress(i / 100.0);
        cv::Mat feature = load_descriptors(test_names[i]);
        cv::Mat featureF;
        feature.convertTo(featureF, CV_32F);
        feature=feature/255.f;
        features.push_back(featureF);
    }

    PRINT_GREEN("[DBoW3::Vocabulary::save] start");
    voc.create(features);
    voc.save("config/test_gftt_10_6.yaml");
    PRINT_GREEN("[DBoW3::Vocabulary::save] end");
}

void testDBowVoc() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  DBoW3::Vocabulary *voc = new DBoW3::Vocabulary();
  voc->load("/home/gvasserm/dev/loop-closure-toolbox/config/mapping_semi_static_ptk_gftt_large_dot_10_6.yaml");
  DBoW3::Database db(*voc, false, 0); // false: do not use direct index (default)

  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");

  std::vector<std::string> test_names;
  std::string test_dir = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/";
  std::vector<std::pair<int, double>> data = read_scores(test_dir + "224.csv");

  std::vector<std::string> extension={".yml"};
  get_files(test_dir, test_names, extension);
  std::sort(test_names.begin(), test_names.end());
  PRINT_YELLOW("[DBoW3::Database::add] start");

  std::map<int, std::string> fnames_map;
  std::map<int, int> id_map;
  for (size_t i = 0; i < test_names.size(); ++i) 
  //for (size_t i=0;i<100; ++i)
  {
    int id =  extract_id(test_names[i]);
    fnames_map[id] = test_names[i];
  }

  for (size_t i = 0; i < data.size(); ++i) {
    printProgress(i / 100.0);
    int ind = data[i].first;
    if (ind < 0){
        continue;
    }
    cv::Mat feature = load_descriptors(fnames_map[ind]);
    db.add(feature);
    id_map[i] = ind;
  }
  
  std::cout << db << std::endl;
  PRINT_GREEN("[DBoW3::Database::add] end\n");

  PRINT_YELLOW("[DBoW3::Database::query] start");

  DBoW3::QueryResults results;
  cv::Mat feature = load_descriptors(fnames_map[224]);
  db.query(feature, results, -1);
  std::cout << "Query results: " << results << std::endl;
  std::sort(results.begin(), results.end(), [](const DBoW3::Result& a, const DBoW3::Result& b) {return a.Id < b.Id;});
  //write_scores_to_csv(results,  id_map, "results.csv");
  PRINT_GREEN("[DBoW3::Database::query] end");
}

void testDBowDatabase() 
{
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
  //for (size_t i=0;i<100; ++i)
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
  //testDBowDatabase();
  //trainVocDesc();
  testDBowVoc();
  return 0;
}
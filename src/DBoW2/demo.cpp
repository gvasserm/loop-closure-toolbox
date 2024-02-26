/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <dirent.h>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

typedef TemplatedVocabulary<FORB::TDescriptor, FORB> ORBVocabulary;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

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

void testDatabase()
{
    std::cout << endl << "Loading ORB Vocabulary. This could take a while..." << std::endl;

    std::vector<std::string> test_names;
    std::vector<std::string> query_names;

    std::string test_dir = "/home/gvasserm/dev/rtabmap/data/samples/";
    std::string query_dir = "/home/gvasserm/dev/rtabmap/data/samples/"; 
    
    get_files(test_dir, test_names);
	  std::sort(test_names.begin(), test_names.end());


    get_files(query_dir, query_names);
	  std::sort(query_names.begin(), query_names.end());
    
    std::string strVocFile = "/home/gvasserm/dev/ORB_SLAM2/Vocabulary/ORBvoc.txt";
    ORBVocabulary *mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);

    if(!bVocLoad)
    {
        std::cerr << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << strVocFile << std::endl;
        std::exit(-1);
    }
    std::cout << "Vocabulary loaded!" << endl << endl;

    OrbDatabase db(*mpVocabulary, false, 0); // false = do not use direct index

    for (size_t i=0;i<test_names.size(); ++i)
    {
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat features;
      loadDetectCompute(test_names[i], keypoints, features);
      vector<cv::Mat> vCurrentDesc = toDescriptorVector(features);
      db.add(vCurrentDesc);
    }
    
    std::cout << "... done!" << std::endl;
    std::cout << "Database information: " << std::endl << db << std::endl;

    // and query the database
    std::cout << "Querying the database: " << std::endl;
    QueryResults ret;
    for (size_t i=0;i<query_names.size(); ++i)
    {
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat features;
      loadDetectCompute(query_names[i], keypoints, features);
      vector<cv::Mat> vCurrentDesc = toDescriptorVector(features);
      db.query(vCurrentDesc, ret, 4);
      std::cout << "Searching for Image " << i << ". " << ret << std::endl;
    }
    
    std::cout << endl;

}

void testVoc()
{

    cv::Mat im1 = cv::imread("/home/gvasserm/data/euroc/MH_01_easy/mav0/cam0/data/1403636579813555456.png", cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread("/home/gvasserm/data/euroc/MH_01_easy/mav0/cam1/data/1403636579763555584.png", cv::IMREAD_GRAYSCALE);

    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat features1, features2;
    
    // int nFeatures = 500;
    // float fScaleFactor = 1.2;
    // int nLevels = 8;
    // int fIniThFAST = 20;
    // int fMinThFAST = 7;

    // ORBextractor *extractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // (*extractor)(im1,cv::Mat(),keypoints1,features1);
    // (*extractor)(im2,cv::Mat(),keypoints2,features2);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    orb->detectAndCompute(im1, cv::Mat(), keypoints1, features1);
    orb->detectAndCompute(im2, cv::Mat(), keypoints2, features2);

    if (false)
    {
        cv::Mat imk1, imk2;
        cv::drawKeypoints(im1, keypoints1, imk1);
        cv::drawKeypoints(im2, keypoints2, imk2);

        // Display the original image and the one with keypoints
        //imshow("Original Image1", im1);
        cv::imshow("Image with Keypoints1", imk1);
        cv::imshow("Image with Keypoints2", imk2);

        cv::waitKey(0);
        // std::cout << features.rows << std::endl;
        // std::cout << s << std::endl;
        // s = vwd->getVisualWords().size();
        // std::cout << s << std::endl;
    }

    vector<cv::Mat> vCurrentDesc1 = toDescriptorVector(features1);
    vector<cv::Mat> vCurrentDesc2 = toDescriptorVector(features2);

    std::cout << endl << "Loading ORB Vocabulary. This could take a while..." << std::endl;

    
    std::string strVocFile = "/home/gvasserm/dev/ORB_SLAM2/Vocabulary/ORBvoc.txt";
    ORBVocabulary *mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        std::cerr << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << strVocFile << std::endl;
        std::exit(-1);
    }
    std::cout << "Vocabulary loaded!" << endl << endl;

    // std::map<WordId, WordValue>
    // WordID - ID od the word
    // WordValue - Value TF-IDF
    DBoW2::BowVector mBowVec1, mBowVec2;
    // std::map<NodeId, std::vector<unsigned int> >
    DBoW2::FeatureVector mFeatVec1, mFeatVec2;

    mpVocabulary->transform(vCurrentDesc1, mBowVec1, mFeatVec1, 4);
    mpVocabulary->transform(vCurrentDesc2, mBowVec2, mFeatVec2, 4);

    std::list<int> words;
    std::vector<int> wordsW;
    for(size_t i=0; i<vCurrentDesc1.size();++i)
    {
        int w = mpVocabulary->transform(vCurrentDesc1[i]);
        words.push_back(w);
        wordsW.push_back(w);
    }

    double s12 = mpVocabulary->score(mBowVec2, mBowVec1);
    double s1 = mpVocabulary->score(mBowVec1, mBowVec1);

    std::cout << s12 << std::endl;
    std::cout << s1 << std::endl;
    return;
}

// ----------------------------------------------------------------------------

int main()
{
  testDatabase();
  //testVoc();

  // vector<vector<cv::Mat > > features;
  // loadFeatures(features);

  // testVocCreation(features);

  // wait();

  // testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  OrbVocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------



#include "DBoW3.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include <dirent.h>

#include "tqdm/tqdm.h"


#include <iostream>
#include <map>

// Define types for better readability
using wordID = int;
using frameID = int;
using score = float;

struct ReferenceScores {
    std::map<wordID, std::map<frameID, score>> reference_scores;
    
    // Function to update the structure with a new frame
    void update(frameID fid, const DBoW3::BowVector& words) {
        for (const auto& word : words) {
            reference_scores[word.first][fid] = word.second;
        }
    }

	const std::map<frameID, score>& findFrames(wordID wid) const {
        static const std::map<frameID, score> emptyMap;  // Static empty map for cases where wordID is not found

        auto it = reference_scores.find(wid);
        if (it != reference_scores.end()) {
            return it->second;
        }
        return emptyMap;
    }
    
    // Function to print the reference_scores map (for debugging)
    void print() const {
        for (const auto& word_pair : reference_scores) {
            std::cout << "wordID " << word_pair.first << ":\n";
            for (const auto& frame_pair : word_pair.second) {
                std::cout << "  frameID " << frame_pair.first << " -> score " << frame_pair.second << "\n";
            }
        }
    }
};


void testDBowDatabase() {

  PRINT_YELLOW("[Loading vocabulary] start");
  std::cout << "Current path is " << fs::current_path() << '\n';
  DBoW3::Vocabulary voc("../../config/sthereo_07_rgb_4_3.yaml");
  DBoW3::Database db(voc, false, 0); // false: do not use direct index (default)
  std::cout << db << std::endl;
  PRINT_GREEN("[Loading vocabulary] end\n");


  PRINT_YELLOW("[DBoW3::Database::add] start");
  auto orb = cv::ORB::create();
  std::vector<std::string> paths = std::move(loadImagePaths("../../assets/01"));
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
    auto db_size = db.add(descriptor) + 1;
  }
  std::cout << db << std::endl;
  PRINT_GREEN("[DBoW3::Database::add] end\n");


  PRINT_YELLOW("[DBoW3::Database::query] start");
  for (auto &path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);

    DBoW3::QueryResults results;
    db.query(descriptor, results, 5);
    std::cout << "Query results: " << results << std::endl;
    break;
  }
  PRINT_GREEN("[DBoW3::Database::query] end");
}


std::map<int, float> computeLikelihoodDBoW(DBoW3::Vocabulary *_vocabulary, 
											const DBoW3::BowVector &key, 										
											const std::vector<DBoW3::BowVector> &queries)
{
	std::map<int, float> likelihood;

	for(int i=0; i<queries.size(); ++i)
	{
		likelihood.insert(likelihood.end(), std::pair<int, float>(i, 0.0f));
	}
	
	// Pour chaque mot dans la signature SURF
	for(int i = 0; i < queries.size(); ++i)
	{
		DBoW3::BowVector query = queries[i];
		if(query.size() > 0){
			float score = _vocabulary->score(key, query);
			likelihood[i] = score;
		}
	}
	return likelihood;
}


std::map<int, float> computeLikelihoodDBoWRef(const DBoW3::BowVector &key,
											const std::vector<int> &ids, 										
											const ReferenceScores &refScores)
{
	std::map<int, float> likelihood;

	for(int i=0; i<ids.size(); ++i)
	{
		likelihood.insert(likelihood.end(), std::pair<int, float>(ids[i], 0.0f));
	}
	
	for(const auto pair: key)
	{
		
		int vw = pair.first;
		const std::map<frameID, score>& refs = refScores.findFrames(vw);
    
    	if (!refs.empty()) {
			for(std::map<int, float>::const_iterator j=refs.begin(); j!=refs.end(); ++j)
			{
				std::map<int, float>::iterator iter = likelihood.find(j->first);
				if(iter != likelihood.end())
				{
					iter->second += j->second;
				}
			}
		}
	}

	return likelihood;
}

void profile_likelihood()
{
	std::string dataset_path = "/home/gvasserm/dev/aicv_amr_ws/results_lc4large_map_def/";
	std::vector<std::string> dataset_files;
	std::vector<std::string> extension = {".yml"};
	get_files(dataset_path, dataset_files, extension);

	std::string strVocFile = "/home/gvasserm/dev/loop-closure-toolbox/config/mapping_ptk_lc4_gftt_10_6.yaml";
	DBoW3::Vocabulary _vocabulary(strVocFile);

	std::vector<DBoW3::BowVector> queries;
	std::vector<int> ids;

	ReferenceScores refScores;

	size_t N = dataset_files.size();
	for(int id : tqdm::range(2000))
	{
		std::string f = dataset_files[id];
		cv::Mat features = load_descriptors(f);
		DBoW3::BowVector bowVector;
		_vocabulary.transform(features, bowVector);
		queries.push_back(bowVector);
		refScores.update(id, bowVector);
		ids.push_back(id);
	}

	// cv::Mat features1 = load_descriptors(dataset_files[0]);
	// DBoW3::BowVector bowVector1;
	// _vocabulary.transform(features1, bowVector1);

	// cv::Mat features2 = load_descriptors(dataset_files[200]);
	// DBoW3::BowVector bowVector2;
	// _vocabulary.transform(features2, bowVector2);

	//std::vector<DBoW3::BowVector> queries(2000, bowVector2);

	DBoW3::BowVector bowVector1 = queries[0];

	auto start_time = std::chrono::high_resolution_clock::now();
	computeLikelihoodDBoW(&_vocabulary, bowVector1, queries);
	auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed_time_ms << " milliseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	computeLikelihoodDBoWRef(bowVector1, ids, refScores);
	end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed_time_ms << " milliseconds" << std::endl;
	return;
}

int main() {
  profile_likelihood();
  //testDBowDatabase();
  return 0;
}
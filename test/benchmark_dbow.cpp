#include <set>
#include <vector>
#include <regex>
#include <opencv2/opencv.hpp>

#include "test_utils.h"
#include "DBoW3.h"

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

    // Function to calculate the size of a nested map in bytes
    size_t calculateMapSize() 
    {
        size_t size = sizeof(reference_scores);

        for (const auto& outerPair : reference_scores) {
            size += sizeof(outerPair.first); // Size of the key (wordID)
            size += sizeof(outerPair.second); // Size of the inner map object

            // Size of the inner map elements
            for (const auto& innerPair : outerPair.second) {
                size += sizeof(innerPair.first); // Size of the key (frameID)
                size += sizeof(innerPair.second); // Size of the value (score)
            }
        }

        return size;
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

std::map<int, float> computeLikelihoodDBoW(const DBoW3::Vocabulary *_vocabulary, 
											const DBoW3::BowVector &key, 										
											const std::map<int, DBoW3::BowVector> &queries)
{
	std::map<int, float> likelihood;

	for(const auto &pair: queries)
	{
		likelihood.insert(likelihood.end(), std::pair<int, float>(pair.first, 0.0f));
	}
	
	// Pour chaque mot dans la signature SURF
	for(const auto &pair: queries)
	{
		DBoW3::BowVector query = pair.second;
		if(query.size() > 0){
			float score = _vocabulary->score(key, query);
			likelihood[pair.first] = score;
		}
	}
	return likelihood;
}

// Function to extract ID from the filename using a regular expression
int extractID(const std::string& filepath) {
    // Extract the filename from the full path
    std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);

    // Regular expression to match the digits following "desc"
    std::regex idRegex(R"(desc(\d+))");
    std::smatch match;

    if (std::regex_search(filename, match, idRegex)) {
        return std::stoi(match.str(1));
    }

    return -1; // Return -1 if no ID is found
}


std::map<int, std::string> files_map_by_ID(const std::vector<std::string> &files)
{
    // Create a vector of pairs to store filenames with their extracted IDs
    std::map<int, std::string> filesWithIDs;

    // Extract IDs and store them along with filenames
    for (const auto& file : files) {
        int id = extractID(file);
        if (id != -1) {
            filesWithIDs[id] = file;
        }
    }

    return filesWithIDs;
}

void load_convert_descriptors(const DBoW3::Vocabulary *voc, const std::map<int, std::string> &filesWithIDs, std::map<int, DBoW3::BowVector> &words_map)
{

    for (const auto& pair : filesWithIDs) {
        cv::Mat features = load_descriptors(pair.second);
        DBoW3::BowVector bowVector;
        voc->transform(features, bowVector);
        words_map[pair.first] = bowVector;
    }
    return;
}

void load_all_descriptors(const DBoW3::Vocabulary *voc, 
    const std::string &dir_path,
    std::map<int, DBoW3::BowVector> &words_map)
{

	std::vector<std::string> dataset_files;
	std::vector<std::string> extension = {".yml"};
	get_files(dir_path, dataset_files, extension);

    std::map<int, std::string> filesWithIDs = files_map_by_ID(dataset_files);
    load_convert_descriptors(voc,filesWithIDs, words_map);

    return;
}

std::map<int, float> query_frame(const DBoW3::Vocabulary *voc, 
                                 const std::string &dir_path, 
                                 std::map<int, DBoW3::BowVector> &words_map,
                                 int kID)
{
    std::string path2file = dir_path + std::to_string(kID) + ".csv";
    std::vector<std::pair<int, float>> scores_gt;
    load_scores(path2file, scores_gt);
    
    if (words_map.find(kID) == words_map.end()) {
        std::cerr << "Key ID " << kID << " not found in words_map." << std::endl;
        return {};
    }
    
    const DBoW3::BowVector &k_vector = words_map.at(kID);
    std::map<int, DBoW3::BowVector> queries;

    std::map<int, float> scores;
    for (const auto& pair : scores_gt) 
    {
        if (words_map.find(pair.first) != words_map.end()) {
            DBoW3::BowVector q_vector = words_map.at(pair.first);
            //float score = voc->score(k_vector, q_vector);
            //scores[pair.first] = score;
            queries[pair.first] = q_vector;

        } else {
            std::cerr << "Query ID " << pair.first << " not found in words_map." << std::endl;
        }
    }

    scores = computeLikelihoodDBoW(voc, k_vector, queries);

    return scores;
}


std::map<int, float> query_frame_ref(const std::string &dir_path, 
                                    std::map<int, DBoW3::BowVector> &words_map,
                                    ReferenceScores &refScores,
                                    int kID)
{
    std::string path2file = dir_path + std::to_string(kID) + ".csv";
    std::vector<std::pair<int, float>> scores_gt;
    load_scores(path2file, scores_gt);
    
    if (words_map.find(kID) == words_map.end()) {
        std::cerr << "Key ID " << kID << " not found in words_map." << std::endl;
        return {};
    }
    
    const DBoW3::BowVector &k_vector = words_map.at(kID);
    std::vector<int> ids;

    std::map<int, float> scores;
    for (const auto& pair : scores_gt) 
    {
        if (words_map.find(pair.first) != words_map.end()) {
            ids.push_back(pair.first);

        } else {
            std::cerr << "Query ID " << pair.first << " not found in words_map." << std::endl;
        }
    }

    scores = computeLikelihoodDBoWRef(k_vector, ids, refScores);

    return scores;
}

// Function to find the key with the maximum value in the map
int findMaxIndex(const std::map<int, float>& scores) {
    if (scores.empty()) {
        throw std::runtime_error("The map is empty.");
    }

    int maxIndex = -1;
    float maxValue = -std::numeric_limits<float>::infinity();

    for (const auto& pair : scores) {
        if (pair.second > maxValue) {
            maxValue = pair.second;
            maxIndex = pair.first;
        }
    }

    return maxIndex;
}

// Function to calculate the precision, recall, and accuracy scores
void calculateMetrics(const std::vector<int>& gt, const std::vector<int>& pred) {
    if (gt.size() != pred.size()) {
        throw std::invalid_argument("Vectors gt and pred must be of the same length.");
    }

    std::set<int> classes(gt.begin(), gt.end());
    classes.insert(pred.begin(), pred.end());

    std::map<int, int> true_positives;
    std::map<int, int> false_positives;
    std::map<int, int> false_negatives;
    std::map<int, int> true_negatives;

    for (int cls : classes) {
        true_positives[cls] = 0;
        false_positives[cls] = 0;
        false_negatives[cls] = 0;
        true_negatives[cls] = 0;
    }

    for (size_t i = 0; i < gt.size(); ++i) {
        for (int cls : classes) {
            if (gt[i] == cls && pred[i] == cls) {
                true_positives[cls]++;
            } else if (gt[i] != cls && pred[i] == cls) {
                false_positives[cls]++;
            } else if (gt[i] == cls && pred[i] != cls) {
                false_negatives[cls]++;
            } else {
                true_negatives[cls]++;
            }
        }
    }

    float macro_precision = 0.0;
    float macro_recall = 0.0;
    for (int cls : classes) {
        int tp = true_positives[cls];
        int fp = false_positives[cls];
        int fn = false_negatives[cls];

        float precision = tp + fp > 0 ? static_cast<float>(tp) / (tp + fp) : 0;
        float recall = tp + fn > 0 ? static_cast<float>(tp) / (tp + fn) : 0;

        macro_precision += precision;
        macro_recall += recall;
    }

    macro_precision /= classes.size();
    macro_recall /= classes.size();

    float accuracy = 0.0;
    for (size_t i = 0; i < gt.size(); ++i) {
        if (gt[i] == pred[i]) {
            accuracy++;
        }
    }
    accuracy /= gt.size();

    std::cout << "Macro Precision: " << macro_precision << std::endl;
    std::cout << "Macro Recall: " << macro_recall << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;
}

// Function to convert bytes to megabytes
double bytesToMB(size_t bytes) {
    return static_cast<double>(bytes) / (1024 * 1024);
}

void benchmark(std::string voc_path, const std::string &dir_path)
{
	DBoW3::Vocabulary vocabulary(voc_path);
    const std::string path2file = dir_path + "loop_closure.csv";
    std::vector<std::pair<int, int>> data;
    load_loop_closures(path2file, data);

    std::map<int, DBoW3::BowVector> words_map;
    load_all_descriptors(&vocabulary, dir_path, words_map);
    ReferenceScores refScores;

    for(const auto &pair : words_map)
    {
        refScores.update(pair.first, pair.second);
    }

    size_t sizeInBytes = refScores.calculateMapSize();
    double sizeInMB = bytesToMB(sizeInBytes);

    std::cout << "Size of nested map in MB: " << sizeInMB << " MB" << std::endl;


    std::vector<int> gt;
    std::vector<int> pred;
    for (size_t i=0;i<data.size();++i) 
    {
        int key = data[i].second;
        int query = data[i].first;
        //std::map<int, float>  pred_scores = query_frame(&vocabulary, dir_path, words_map, key);
        std::map<int, float>  pred_scores = query_frame_ref(dir_path, words_map, refScores, key);
        int maxIndex = findMaxIndex(pred_scores);
        gt.push_back(query);
        pred.push_back(maxIndex);
    }

    calculateMetrics(gt, pred);
    
    return;
}

int main() {
    std::string voc_path = "config/mapping_ptk_lc4_gftt_10_6.yaml";
    std::string dir_path = "/home/gvasserm/data/AMRLoopClosureData/None-warehouse_PTK-4_D455f_Cameras-static_20231121_132141692/";
    benchmark(voc_path, dir_path);
    return 0;
}
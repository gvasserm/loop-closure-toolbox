#include "DBoW3.h"
//#include "DBoW2.h"
#include "VLAD.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind_utils.hpp"

namespace py = pybind11;

namespace loopclosuretoolbox {

PYBIND11_MODULE(loopclosuretoolbox, m) {
  // Import the submodules
  py::module m_dbow = m.def_submodule("dbow");
  //py::module m_dbow2 = m.def_submodule("dbow2");
  py::module m_vlad = m.def_submodule("vlad");
  
  /*
    //----------------------DBoW2---------------------------
  py::enum_<DBoW2::WeightingType>(m_dbow2, "WeightingType")
      .value("TF_IDF", DBoW2::WeightingType::TF_IDF)
      .value("TF", DBoW2::WeightingType::TF)
      .value("IDF", DBoW2::WeightingType::IDF)
      .value("BINARY", DBoW2::WeightingType::BINARY);

  py::enum_<DBoW2::ScoringType>(m_dbow2, "ScoringType")
      .value("L1_NORM", DBoW2::ScoringType::L1_NORM)
      .value("L2_NORM", DBoW2::ScoringType::L2_NORM)
      .value("CHI_SQUARE", DBoW2::ScoringType::CHI_SQUARE)
      .value("KL", DBoW2::ScoringType::KL)
      .value("BHATTACHARYYA", DBoW2::ScoringType::BHATTACHARYYA)
      .value("DOT_PRODUCT", DBoW2::ScoringType::DOT_PRODUCT);pip install -e
  
  vocab_dbow2.def(
      "create",
      [](DBoW2::TemplatedVocabulary &self, const py::list &list_of_ndarray) {
        std::vector<cv::Mat> features;
        for (const auto &feature : list_of_ndarray) {
          py::array_t<uint8_t> feature_casted = feature.cast<py::array_t<uint8_t>>();
          cv::Mat feature_mat = toMat<uint8_t>(feature_casted);
          features.push_back(feature_mat);
        }
        self.create(features);
      },
      py::arg("training_features"));
  vocab_dbow2.def(
      "save",
      [](DBoW2::TemplatedVocabulary &self, const std::string &filename, bool binary) {
        self.save(filename, binary);
      },
      py::arg("filename"), py::arg("binary") = true);
  vocab_dbow2.def(
      "save_txt",
      [](DBoW2::TemplatedVocabulary &self, const std::string &filename) {
        self.saveToTextFile(filename);
      },
      py::arg("filename"));
  vocab_dbow2.def("size", &DBoW2::TemplatedVocabulary::size);
  vocab_dbow2.def(
      "get_word_weight",
      [](DBoW2::TemplatedVocabulary &self, unsigned int word_id) {
        double word_weight = self.getWordWeight(word_id);
        return word_weight;
      },
      py::arg("word_id"));
  vocab_dbow2.def(
      "get_word_descriptor",
      [](DBoW2::TemplatedVocabulary &self, unsigned int word_id) {
        cv::Mat word_descriptor_mat = self.getWord(word_id);
        auto word_descriptor_array = toArray<uint8_t>(word_descriptor_mat);
        return word_descriptor_array;
      },
      py::arg("word_id"));
  vocab_dbow2.def(
    "transform",
    [](DBoW2::TemplatedVocabulary &self, py::array_t<uint8_t> &features) {
      cv::Mat mat = toMat<uint8_t>(features);
      DBoW3::BowVector bow_vector;
      self.transform(mat, bow_vector);
      // convert bow_vector to pair of (key, value)
      py::list py_bow_vector;
      for (const auto &key_value : bow_vector) {
        py_bow_vector.append(py::make_tuple(key_value.first, key_value.second));
      }
      return py_bow_vector;
    },
    py::arg("features"));
  vocab_dbow2.def("__repr__", [](DBoW2::TemplatedVocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab_dbow2.def("__str__", [](DBoW2::TemplatedVocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<DBoW2::TemplatedDatabase> db_dbow2(m_dbow2, "Database");
  db_dbow2.def(py::init<const DBoW2::TemplatedVocabulary &, bool, int>(), py::arg("voc"),
              py::arg("use_di") = true, py::arg("di_levels") = 0);
  db_dbow2.def(
      "add",
      [](DBoW2::TemplatedDatabase &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db_dbow2.def(
      "query",
      [](DBoW2::TemplatedDatabase &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        cv::Mat mat = toMat<uint8_t>(features);
        DBoW2::QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.Id, result.Score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db_dbow2.def("compute_pairwise_score", [](DBoW2::TemplatedDatabase &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db_dbow2.def("size", &DBoW2::TemplatedDatabase::size);
  db_dbow2.def("__repr__", [](DBoW2::TemplatedDatabase &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db_dbow2.def("__str__", [](DBoW2::TemplatedDatabase &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  */

  //----------------------DBoW3---------------------------
  
  py::enum_<DBoW3::WeightingType>(m_dbow, "WeightingType")
      .value("TF_IDF", DBoW3::WeightingType::TF_IDF)
      .value("TF", DBoW3::WeightingType::TF)
      .value("IDF", DBoW3::WeightingType::IDF)
      .value("BINARY", DBoW3::WeightingType::BINARY);

  py::enum_<DBoW3::ScoringType>(m_dbow, "ScoringType")
      .value("L1_NORM", DBoW3::ScoringType::L1_NORM)
      .value("L2_NORM", DBoW3::ScoringType::L2_NORM)
      .value("CHI_SQUARE", DBoW3::ScoringType::CHI_SQUARE)
      .value("KL", DBoW3::ScoringType::KL)
      .value("BHATTACHARYYA", DBoW3::ScoringType::BHATTACHARYYA)
      .value("DOT_PRODUCT", DBoW3::ScoringType::DOT_PRODUCT);

  py::class_<DBoW3::Vocabulary> vocab_dbow(m_dbow, "Vocabulary");
  vocab_dbow.def(py::init<int, int>());            // k, L
  vocab_dbow.def(py::init<int, int, DBoW3::WeightingType, DBoW3::ScoringType>()); // k, L, weighting, scoring
  vocab_dbow.def(py::init<const std::string &>()); // load from file
  vocab_dbow.def(
      "create",
      [](DBoW3::Vocabulary &self, const py::list &list_of_ndarray) {
        std::vector<cv::Mat> features;
        for (const auto &feature : list_of_ndarray) {
          py::array_t<uint8_t> feature_casted = feature.cast<py::array_t<uint8_t>>();
          cv::Mat feature_mat = toMat<uint8_t>(feature_casted);
          features.push_back(feature_mat);
        }
        self.create(features);
      },
      py::arg("training_features"));

  vocab_dbow.def(
      "load",
      [](DBoW3::Vocabulary &self, const std::string &filename) {
        self.load(filename);
      },
      py::arg("filename"));
  
  vocab_dbow.def(
      "save",
      [](DBoW3::Vocabulary &self, const std::string &filename, bool binary) {
        self.save(filename, binary);
      },
      py::arg("filename"), py::arg("binary") = true);
  vocab_dbow.def(
      "save_txt",
      [](DBoW3::Vocabulary &self, const std::string &filename) {
        self.saveToTextFile(filename);
      },
      py::arg("filename"));
  vocab_dbow.def("size", &DBoW3::Vocabulary::size);
  vocab_dbow.def(
      "get_word_weight",
      [](DBoW3::Vocabulary &self, unsigned int word_id) {
        double word_weight = self.getWordWeight(word_id);
        return word_weight;
      },
      py::arg("word_id"));
  vocab_dbow.def(
      "get_word_descriptor",
      [](DBoW3::Vocabulary &self, unsigned int word_id) {
        cv::Mat word_descriptor_mat = self.getWord(word_id);
        auto word_descriptor_array = toArray<uint8_t>(word_descriptor_mat);
        return word_descriptor_array;
      },
      py::arg("word_id"));
  vocab_dbow.def(
    "transform",
    [](DBoW3::Vocabulary &self, py::array_t<uint8_t> &features) {
      cv::Mat mat = toMat<uint8_t>(features);
      DBoW3::BowVector bow_vector;
      self.transform(mat, bow_vector);
      // convert bow_vector to pair of (key, value)
      py::list py_bow_vector;
      for (const auto &key_value : bow_vector) {
        py_bow_vector.append(py::make_tuple(key_value.first, key_value.second));
      }
      return py_bow_vector;
    },
    py::arg("features"));
  // vocab_dbow.def(
  //   "score",
  //   [](DBoW3::Vocabulary &self, py::list &py_bow_vector1, py::list &py_bow_vector2) {
  //     DBoW3::BowVector bow_vector1;
  //     DBoW3::BowVector bow_vector2;
  //     float score = self.score(bow_vector1, bow_vector1);
  //     // convert bow_vector to pair of (key, value)
  //     py::list py_bow_vector;
  //     for (const auto &key_value : py_bow_vector1) {
  //       py_bow_vector.append(py::make_tuple(key_value.first, key_value.second));
  //     }
  //     return py_bow_vector;
  //   },
  //   py::arg("score"));
  vocab_dbow.def("__repr__", [](DBoW3::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab_dbow.def("__str__", [](DBoW3::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<DBoW3::Database> db_dbow(m_dbow, "Database");
  db_dbow.def(py::init<const DBoW3::Vocabulary &, bool, int>(), py::arg("voc"),
              py::arg("use_di") = true, py::arg("di_levels") = 0);
  db_dbow.def(
      "add",
      [](DBoW3::Database &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db_dbow.def(
      "query",
      [](DBoW3::Database &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        cv::Mat mat = toMat<uint8_t>(features);
        DBoW3::QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.Id, result.Score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db_dbow.def("compute_pairwise_score", [](DBoW3::Database &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db_dbow.def("size", &DBoW3::Database::size);
  db_dbow.def("__repr__", [](DBoW3::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db_dbow.def("__str__", [](DBoW3::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  //----------------------VLAD-------------------------------

  py::class_<VLAD::Vocabulary> vocab_vlad(m_vlad, "Vocabulary");
  vocab_vlad.def(py::init<const std::string &>()); // load from file
  vocab_vlad.def("size", &VLAD::Vocabulary::size);
  vocab_vlad.def("__repr__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  vocab_vlad.def("__str__", [](VLAD::Vocabulary &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });

  py::class_<VLAD::Database> db_vlad(m_vlad, "Database");
  db_vlad.def(py::init<const std::string &>());
  db_vlad.def(
      "add",
      [](VLAD::Database &self, py::array_t<uint8_t> &features) {
        cv::Mat mat = toMat<uint8_t>(features);
        auto entry_id = self.add(mat);
        return entry_id;
      },
      py::arg("features"));
  db_vlad.def(
      "query",
      [](VLAD::Database &self, py::array_t<uint8_t> &features, int max_results, int max_id) {
        cv::Mat mat = toMat<uint8_t>(features);
        VLAD::QueryResults results;
        self.query(mat, results, max_results, max_id);

        py::list py_results;
        for (const auto &result : results) {
          py_results.append(py::make_tuple(result.id, result.score));
        }
        return py_results;
      },
      py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1);
  db_vlad.def("compute_pairwise_score", [](VLAD::Database &self) {
    cv::Mat pscore_mat = self.computepairwiseScore();
    auto pscore_array = toArray<double>(pscore_mat);
    return pscore_array;
  });
  db_vlad.def("size", &VLAD::Database::size);
  db_vlad.def("__repr__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  db_vlad.def("__str__", [](VLAD::Database &self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
}

} // namespace loopclosuretoolbox
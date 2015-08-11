nclude <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

using namespace boost::filesystem;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

std::vector<std::string> find_img_paths(std::string root_path, boost::regex ext = boost::regex(".(jpg|JPG)$")){
  std::vector<string> img_paths;
  path current_dir(root_path); //
  for (recursive_directory_iterator iter(current_dir), end;
       iter != end;
       ++iter)
  {
    std::string name = iter->path().filename().string();
    if (regex_search(name, ext))
      img_paths.push_back(iter->path().string());
  }
  std::cout << "Number of img files -- " << img_paths.size()<< std::endl;
  return img_paths;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}


template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_to_csv input_data_root_path output_file_path  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  arg_pos = 0;

  // Get the list of img_files
  std::vector<std::string> img_paths;
  img_paths = find_img_paths(argv[++arg_pos]);
  LOG(INFO) << "Root paths for images -- " << argv[arg_pos];

  // Open output file stream
  // Create CSV file
  std::ofstream csv_file(argv[++arg_pos]);
  LOG(INFO) << "CSV file path -- " << argv[arg_pos];

  // Define the Net Proto
  std::string feature_extraction_proto(argv[++arg_pos]);
  LOG(INFO) << "Model definition file path -- " << argv[arg_pos];

  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);
  LOG(INFO) <<  "Pretrained Binary Proto path -- "<< argv[arg_pos];

  // Get Blob name to write in csv
  std::string blob_name;
  blob_name = argv[++arg_pos];
  LOG(INFO) << "Layer name -- " << argv[arg_pos];


  //get the net
  Net<float> feature_extraction_net(feature_extraction_proto, caffe::TEST);

  // Check the given blob name
  CHECK(feature_extraction_net.has_blob(blob_name))
        << "Unknown feature blob name " << blob_name
        << " in the network " << feature_extraction_proto;

  //get trained net
  feature_extraction_net.CopyTrainedLayersFrom(pretrained_binary_proto);


  // Define data layer
  boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_data_layer =
     boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(
      		feature_extraction_net.layer_by_name("data"));


  int batch_size = memory_data_layer->batch_size();
  //int num_iters = img_paths.size() / batch_size;
  int img_rem = img_paths.size() % batch_size;

  for (int batch_indx = 0; batch_indx < img_paths.size(); batch_indx += batch_size){
  	  // Load Images
  		std::vector<cv::Mat> images(batch_size);
  		std::vector<int> labels(batch_size, 0);
  		for(int i  = 0; i < batch_size; i++){
  			bool is_color;
  			is_color = (memory_data_layer->channels() == 1) ? is_color = 0 : is_color = 1;
  			cv::Mat image = ReadImageToCVMat(img_paths[batch_indx+i], memory_data_layer->height(),
  					memory_data_layer->width(), is_color);
  			images[0] = image;
  		}

  		CHECK_EQ(images.size(), batch_size) << "Number of images and image paths are not equal!";
  	  memory_data_layer->AddMatVector(images, labels);

  	  // Run the Net
  	  float loss;
  	  const std::vector<Blob<float>*>& result = feature_extraction_net.ForwardPrefilled(&loss);

  	  boost::shared_ptr<Blob<float> > target_blob = feature_extraction_net.blob_by_name(blob_name);

  	  LOG(INFO)<< "Output result size: "<< target_blob->num();
  	  // Now result will contain the argmax results.
  	  const float* values = target_blob->cpu_data();
  	  for (int i = 0; i < result[1]->num(); ++i) {
  	    LOG(INFO)<< " Image: "<< i << " class:" << values[i];
  }


  }


  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}


int main(int argc, char** argv) {
	// /media/retina18/66f4f5c6-ed98-470a-978f-f667aed46a88/KAGGLE/train features.csv /home/retina18/Downloads/caffe_dev/models/kaggle/maxout_net_v1_mem_data_layer.prototxt /media/retina18/66f4f5c6-ed98-470a-978f-f667aed46a88/KAGGLE/CAFFE_STUFF_2/MODELS/maxout_net_rmsprop_iter_100000.caffemodel prob GPU;
	return feature_extraction_pipeline<float>(argc, argv);
  //return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
	//std::vector<std::string> img_paths;
	//img_paths = find_img_paths("/media/retina18/66f4f5c6-ed98-470a-978f-f667aed46a88/KAGGLE/train");

}

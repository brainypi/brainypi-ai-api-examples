/**
 *
 * @brief      This file implements api client example images.
 *
 * @author     ShunyaOS Team
 * @date       2023
 */
#include <pistache/client.h>
#include <pistache/http.h>
#include <pistache/net.h>

#include <filesystem>
#include <iostream>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

using namespace Pistache;
using namespace std;
/**
 * @brief      send data to the API endpoint
 *
 * @param      frame      - input image
 * @param      client     - HTTP client object
 * @param      url        - API endpoint URL
 * @param      responses  - vector of promises for HTTP responses
 *
 * @return     JSON response from the API
 */
std::string send_request_to_api_server(
	cv::Mat &frame, Http::Experimental::Client &client, string &url,
	std::vector<Async::Promise<Http::Response> > &responses);

/**
 * @brief      send data to the API endpoint
 *
 * @param      input      - input json
 * @param      client     - HTTP client object
 * @param      url        - API endpoint URL
 * @param      responses  - vector of promises for HTTP responses
 *
 * @return     JSON response from the API
 */
std::string send_json_request_to_api_server(
	std::string &input, Http::Experimental::Client &client, string &url,
	std::vector<Async::Promise<Http::Response> > &responses);

/**
 * @brief      Draw a label on an input image with the specified text, position
 *             and color.
 *
 * @param      input_image  The input image on which the label is to be drawn.
 * @param      label        The text to be displayed as the label.
 * @param      left         The x-coordinate of the top-left corner of the
 *                          bounding box.
 * @param      top          The y-coordinate of the top-left corner of the
 *                          bounding box.
 */
void draw_label(cv::Mat &input_image, string label, int left, int top);

void draw_bounding_box(cv::Mat &input_image, int left, int top, int width, int height);

/**
 * @brief      Dispaly the image 
 *
 * @param      image  The image
 */
void display_output_image(cv::Mat &image);

/**
 * @brief      Saves image to disk to disk.
 *
 * @param      image       The image
 * @param[in]  image_path  The image path
 * @param[in]  output_dir  The output dir
 */
void save_to_disk(cv::Mat &image, const std::string image_path,
		  const std::string output_dir);

/**
 * @brief      Saves face embeddings to JSON file on disk.
 *
 * @param[in]  name	   name of the person
 * @param[in]  embeddings  The face embeddings
 * @param[in]  output_dir  The output dir
 */
void save_embeddings_to_disk(const std::string &name, 
			     const rapidjson::Value &embeddings, 
			     const std::string &output_dir,
			     const std::string &json_path);
/**
 * @brief      This file implements api client example images.
 *
 * @author     ShunyaOS Team
 * @date       2023
 */

#include <iostream>
#include <filesystem>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <pistache/client.h>
#include <pistache/http.h>
#include <pistache/net.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

#include "helper.hpp"

#define MIN_FACE_DET_CONFIDENCE 0.5f

using namespace Pistache;
using namespace std;

/**
 * @brief      Processes the input images and detects objects in them.
 *
 * @param      url        - The URL of the API server
 * @param      image_dir  - The directory where the input images are stored.
 * @param      out_dir    - The directory where the output images with objects
 *                        detected will be saved.
 * @param      save       - Boolean indicating if the output images with objects
 *                        detected will be saved or not.
 *
 * @return     void
 */
void register_face(std::string &url, std::string &image_path, 
			 const std::string out_dir, const bool save, 
			 const bool display, const std::string &name)
{
	rapidjson::Document output_json;
	Http::Experimental::Client client;
	std::vector<Async::Promise<Http::Response> > responses;
	auto opts = Http::Experimental::Client::options()
			    .threads(8)
			    .maxConnectionsPerHost(1)
			    .maxResponseSize(1024 * 1024 * 100);

	client.init(opts);

	cv::Mat image = cv::imread(image_path);

	std::string result =
		send_request_to_api_server(image, client, url, responses);

	if (output_json.Parse(result.c_str()).HasParseError()) {
		std::cerr
			<< "Error: Failed to parse JSON output from API server. "
			<< result << std::endl;
		return;
	}

	if (output_json.HasMember("error")) {
		// Print error and return
		std::cerr << "Error: Server returned error\nCode: "
			  << output_json["error"]["code"].GetInt()
			  << "\nReason: "
			  << output_json["error"]["message"].GetString()
			  << std::endl;
		return;
	}

	if (output_json["result"]["faces"].Size() < 1) {
		std::cerr << "Error: No face Detected in input image."
			  << std::endl;
		return;
	}

        std::cout << result<< std::endl;
	cv::Mat frame = image.clone();

	for (auto i = 0; i < output_json["result"]["faces"].Size(); i++) {
		float confidence =
			output_json["result"]["faces"][i]["confidence"]
				.GetFloat();
		/* Check if the confidence is above threshold */
		if (confidence < MIN_FACE_DET_CONFIDENCE) {
			continue;
		}
		int top = (int)output_json["result"]["faces"][i]["boundingBox"]
					  ["top"]
						  .GetFloat();

		int left = (int)output_json["result"]["faces"][i]["boundingBox"]
					   ["left"]
						   .GetFloat();

		int width = (int)output_json["result"]["faces"][i]
					    ["boundingBox"]["width"]
						    .GetFloat();

		int height = (int)output_json["result"]["faces"][i]
					     ["boundingBox"]["height"]
						     .GetFloat();

		std::string label = name + std::to_string(i + 1) + " " +
				    std::to_string(confidence);

		draw_bounding_box(frame, left, top, width, height);

		/* Uncomment if you want to draw labels */
		//draw_label(frame, label, left, top);
	
		/* Save embeddings to disk */
		save_embeddings_to_disk(name, 
				output_json["result"]["faces"][i]["embeddings"], 
				out_dir, "face_embeddings.json");
	}

	if (display) {
		display_output_image(frame);
	}
	if (save) {
		save_to_disk(frame, image_path, out_dir);
	}

	client.shutdown();
}

int main(int argc, char **argv)
{
	std::string url = "http://localhost:9900/v1/face2embedding";
	std::string input_img = "../sample_inputs/images/face.jpg";
	std::string output_dir = "./output";
	bool save = true;
	bool display = true;
	std::string name = "Person1";	
	std::cout << "Starting client..." << std::endl;
	register_face(url, input_img, output_dir, save, display, name);

	return 0;
}

/**
 * @brief      This file implements api client example images.
 *
 * @author     ShunyaOS Team
 * @date       2023
 */

#include <iostream>
#include <filesystem>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

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

#define MIN_OBJ_DET_CONFIDENCE 0.5f

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
void detect_objects(std::string &url, std::string &image_path,
		    const std::string out_dir, const bool save,
		    const bool display)
{
	rapidjson::Document output_json;
	Http::Experimental::Client client;
	std::vector<Async::Promise<Http::Response> > responses;
	auto opts = Http::Experimental::Client::options()
			    .threads(8)
			    .maxConnectionsPerHost(1)
			    .maxResponseSize(1024 * 1024 * 100);

	client.init(opts);

	// Read the image using OpenCV
	cv::Mat image = cv::imread(image_path);

	// Send the image as a request to the specified web page
	std::string result =
		send_request_to_api_server(image, client, url, responses);

	if (output_json.Parse(result.c_str()).HasParseError())
		std::cerr
			<< "Error: Failed to parse JSON output from API server. "
			<< result << std::endl;

	if (output_json.HasMember("error")) {
		// Print error and return
		std::cerr << "Error: Server returned error\nCode: "
			  << output_json["error"]["code"].GetInt()
			  << "\nReason: "
			  << output_json["error"]["message"].GetString()
			  << std::endl;
		return;
	}

	if (output_json["result"]["objects"].Size() < 1) {
		std::cerr << "Error: No objects Detected in input image."
			  << std::endl;
		return;
	}

	cv::Mat frame = image.clone();
	for (auto i = 0; i < output_json["result"]["objects"].Size(); i++) {
		float confidence =
			output_json["result"]["objects"][i]["confidence"]
				.GetFloat();
		/*Check if the confidence is above threshold*/
		if (confidence < MIN_OBJ_DET_CONFIDENCE) {
			continue;
		}
		int top = (int)output_json["result"]["objects"][i]
					  ["boundingBox"]["top"]
						  .GetFloat();

		int left = (int)output_json["result"]["objects"][i]
					   ["boundingBox"]["left"]
						   .GetFloat();

		int width = (int)output_json["result"]["objects"][i]
					    ["boundingBox"]["width"]
						    .GetFloat();

		int height = (int)output_json["result"]["objects"][i]
					     ["boundingBox"]["height"]
						     .GetFloat();

		std::string obj = output_json["result"]["objects"][i]["object"]
					  .GetString();
		string label = obj + " " + std::to_string(confidence);
		draw_bounding_box(frame, left, top, width, height);
		draw_label(frame, label, left, top);
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
	std::string url = "http://localhost:9900/v1/detectobjects";
	std::string input_img = "../sample_inputs/images/car.jpg";
	std::string output_dir = "./output";
	bool save = true;
	bool display = true;

	cout << "Starting client...\n";
	detect_objects(url, input_img, output_dir, save, display);

	return 0;
}

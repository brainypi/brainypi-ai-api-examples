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

#define EMBEDDINGS_DB "face_embeddings.json"

using namespace Pistache;
using namespace std;

/**
 * @brief  
 * 
 * @param url 
 * @param embeddings1 
 * @param embeddings2 
 * @return float 
 */
float compare_face(const std::string &url, rapidjson::Value &embeddings1,
		   rapidjson::Value &embeddings2)
{
	Http::Experimental::Client client;
	std::vector<Async::Promise<Http::Response> > responses;
	auto opts = Http::Experimental::Client::options()
			    .threads(8)
			    .maxConnectionsPerHost(1)
			    .maxResponseSize(1024 * 1024 * 100);

	client.init(opts);

	std::string url2 = url + "/v1/compareface";

	/* Convert the embeddings into input json */
	rapidjson::Document input_json;
	rapidjson::Document output_json;
	rapidjson::Value face1(rapidjson::kObjectType);
	rapidjson::Value face2(rapidjson::kObjectType);

	input_json.SetObject();

        rapidjson::Value embeddings1Copy(embeddings1,
					input_json.GetAllocator());

        rapidjson::Value embeddings2Copy(embeddings2,
					input_json.GetAllocator());

	face1.AddMember("embeddings", embeddings1Copy, input_json.GetAllocator());
	face2.AddMember("embeddings", embeddings2Copy, input_json.GetAllocator());

	input_json.AddMember("face1", face1, input_json.GetAllocator());
	input_json.AddMember("face2", face2, input_json.GetAllocator());

	/* Convert JSON to string */
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	writer.SetMaxDecimalPlaces(2);
	input_json.Accept(writer);

	std::string input_json_str(buffer.GetString(), buffer.GetSize());

	std::string result = send_json_request_to_api_server(
		input_json_str, client, url2, responses);

	if (output_json.Parse(result.c_str()).HasParseError()) {
		std::cerr
			<< "Error: Failed to parse JSON output from API server. "
			<< result << std::endl;
		return 0;
	}

	if (output_json.HasMember("error")) {
		// Print error and return
		std::cerr << "Error: Server returned error\nCode: "
			  << output_json["error"]["code"].GetInt()
			  << "\nReason: "
			  << output_json["error"]["message"].GetString()
			  << std::endl;
		return 0;
	}

        float out = output_json["result"]["confidence"].GetFloat();

        client.shutdown();
	return out * 10;
}

/**
 * @brief 
 * 
 * @param url 
 * @param input_embeddings 
 * @param output_dir 
 * @param json_path 
 * @return std::string 
 */
std::string find_face(const std::string &url, rapidjson::Value &input_embeddings,
		      const std::string &output_dir,
		      const std::string &json_path)
{
	rapidjson::Document embedings_db;
	std::string name = "Unknown";

	/* Check if the file exists */
	std::string output_file = output_dir + json_path;

	if (!filesystem::exists(output_dir)) {
		/* Create directory */
		filesystem::create_directory(output_dir);
	}
	if (!filesystem::exists(output_file)) {
		return name;
	}

	/* Read existing db file */
	std::ifstream ifs(output_file);
	rapidjson::IStreamWrapper isw(ifs);

	embedings_db.ParseStream(isw);

	/* Compare face embeddings */
	for (auto i = 0; i < embedings_db.Size(); i++) {
		float confidence = compare_face(url, input_embeddings,
						embedings_db[i]["embeddings"]);

		if (confidence > 0.8f) {
			/* Face found */
			name = embedings_db[i]["name"].GetString();
		}
	}

	return name;
}

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
void verify_face(const std::string &url, std::string &image_path,
		 const std::string out_dir, const bool save, const bool display)
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

	std::string url2 = url + "/v1/face2embedding";

	std::string result =
		send_request_to_api_server(image, client, url2, responses);

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

		draw_bounding_box(frame, left, top, width, height);

		std::string name = find_face(
			url, output_json["result"]["faces"][i]["embeddings"],
			out_dir, EMBEDDINGS_DB);
		std::string label = name + std::to_string(i + 1) + " " +
				    std::to_string(confidence);
		/* Uncomment if you want to draw labels */
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
	std::string url = "http://localhost:9900";
	std::string input_img = "../sample_inputs/images/faces.jpg";
	std::string output_dir = "./output";
	bool save = true;
	bool display = true;
	std::cout << "Starting client..." << std::endl;
	verify_face(url, input_img, output_dir, save, display);

	return 0;
}

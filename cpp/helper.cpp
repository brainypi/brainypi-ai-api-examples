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
	std::vector<Async::Promise<Http::Response> > &responses)
{
	// Encode the input frame as a jpg image
	std::vector<uchar> buf;
	cv::imencode(".jpg", frame, buf);
	std::string image_data(buf.begin(), buf.end());

	// Clear the buffer and shrink it to fit
	buf.clear();
	buf.shrink_to_fit();
	// Send the image data as a post request to the API endpoint
	auto resp = client.post(url).body(image_data).send();

	// Store the JSON response from the API
	std::string result;

	// Handle the response from the API
	resp.then(
		[&](Http::Response response) {
			std::cout << "Response code = " << response.code()
				  << std::endl;
			auto body = response.body();
			if (!body.empty()) {
				std::cout << "Response body size = "
					  << body.size() << std::endl;
				// Handle response
				result = body;
			}
		},
		[&](std::exception_ptr
			    exc) { // In case of an exception, set the objects count to 0
			result = "";
			PrintException excPrinter;
			excPrinter(exc);
		});
	// Push the response promise to the vector of responses
	responses.push_back(std::move(resp));

	// Wait for all the responses to complete
	auto sync = Async::whenAll(responses.begin(), responses.end());
	Async::Barrier<std::vector<Http::Response> > barrier(sync);
	barrier.wait();

	// Return the JSON response from the API
	return result;
}

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
	std::vector<Async::Promise<Http::Response> > &responses)
{
	// Send the image data as a post request to the API endpoint
	auto resp = client.post(url).body(input).send();

	// Store the JSON response from the API
	std::string result;

	// Handle the response from the API
	resp.then(
		[&](Http::Response response) {
			std::cout << "Response code = " << response.code()
				  << std::endl;
			auto body = response.body();
			if (!body.empty()) {
				std::cout << "Response body size = "
					  << body.size() << std::endl;
				// Handle response
				result = body;
			}
		},
		[&](std::exception_ptr
			    exc) { // In case of an exception, set the objects count to 0
			result = "";
			PrintException excPrinter;
			excPrinter(exc);
		});
	// Push the response promise to the vector of responses
	responses.push_back(std::move(resp));

	// Wait for all the responses to complete
	auto sync = Async::whenAll(responses.begin(), responses.end());
	Async::Barrier<std::vector<Http::Response> > barrier(sync);
	barrier.wait();

	// Return the JSON response from the API
	return result;
}

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
void draw_label(cv::Mat &input_image, string label, int left, int top)
{
	// Display the label at the top of the bounding box.
	int baseLine;
	cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
					      1, 1, &baseLine);
	top = max(top, label_size.height);
	// Top left corner.
	cv::Point tlc = cv::Point(top, left);
	// Bottom right corner.
	cv::Point brc = cv::Point(top + label_size.width,
				  left + label_size.height + baseLine);
	// Draw white rectangle.
	cv::rectangle(input_image, tlc, brc, cv::Scalar(0, 0, 0), cv::FILLED);
	// Put the label on the black rectangle.
	cv::putText(input_image, label,
		    cv::Point(top, left + label_size.height),
		    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 1);
}

void draw_bounding_box(cv::Mat &input_image, int left, int top, int width,
		       int height)
{
	// Draw bounding boxes
	cv::rectangle(input_image, cv::Rect2i(top, left, width, height),
		      cv::Scalar(0, 255, 255), 2);
}

/**
 * @brief      Dispaly the image 
 *
 * @param      image  The image
 */
void display_output_image(cv::Mat &image)
{
	// Visualize results
	std::cout << "Press any key to continue..." << std::endl;
	cv::imshow("Result Image", image);
	cv::waitKey(0);
}

/**
 * @brief      Saves image to disk to disk.
 *
 * @param      image       The image
 * @param[in]  image_path  The image path
 * @param[in]  output_dir  The output dir
 */
void save_to_disk(cv::Mat &image, const std::string image_path,
		  const std::string output_dir)
{
	cout << "Saving results...\n";
	if (!filesystem::exists(output_dir)) {
		filesystem::create_directory(output_dir);
	}
	// Save the output image
	std::string output_image =
		output_dir + "/result_" +
		image_path.substr(image_path.find_last_of("/") + 1);
	cv::imwrite(output_image, image);
}

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
			     const std::string &json_path)
{
	std::fstream file;
	rapidjson::Document embedings_json;
	rapidjson::Value face(rapidjson::kObjectType);
	/* Check if the file exists */
	std::string output_file = output_dir + "/" + json_path;

	if (!filesystem::exists(output_dir)) {
		/* Create directory */
		filesystem::create_directory(output_dir);
	}
	if (filesystem::exists(output_file)) {
		/* Read existing file */
		std::ifstream ifs(output_file);
		rapidjson::IStreamWrapper isw(ifs);

		embedings_json.ParseStream(isw);
	} else {
		file.open(output_file, ios::out);
		if (!file) {
			std::cerr << "Error in creating file!!!"<<std::endl;
			file.close();
                        return;
		}
                file.close();
		embedings_json.SetArray();
	}

	/* Add member */
	rapidjson::Value nameValue(name.c_str(), embedings_json.GetAllocator());
	face.AddMember("name", nameValue, embedings_json.GetAllocator());
	rapidjson::Value embeddingsCopy(embeddings,
					embedings_json.GetAllocator());
	face.AddMember("embeddings", embeddingsCopy,
		       embedings_json.GetAllocator());
	embedings_json.PushBack(face, embedings_json.GetAllocator());

	/* Save to Disk */
	std::cout << "Saving results..." << std::endl;
	std::ofstream ofs(output_file);
	rapidjson::OStreamWrapper osw(ofs);

	rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
	embedings_json.Accept(writer);
}

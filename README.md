# BrainyPi AI API Examples

Welcome to the BrainyPi AI API Examples repository! This repository provides you with a collection of example applications that demonstrate the capabilities of the BrainyPi AI REST server. These examples cover various computer vision tasks, including face detection, face recognition, object detection, pose estimation, and image classification.

## Installation

To get started, make sure you have the BrainyPi AI REST server installed. If you haven't installed it yet, 
run the command on BrainyPi running Rbian v0.7.3-beta.
```sh
sudo apt install brainypi-ai-server
```

Once the BrainyPi AI REST server is up and running, follow these steps to install and run the examples:

1. Clone this repository to your local machine:

```sh
git clone https://github.com/brainypi/brainypi-ai-api-examples.git
```

2. Install the necessary dependencies:

```sh
sudo apt install rapidjson-dev
```

## Running Examples

The examples in this repository are designed to showcase different computer vision tasks supported by the BrainyPi AI REST server. To run the examples, follow these steps:

1. Navigate to the root directory of the cloned repository:

```sh
cd brainypi-ai-api-examples
```

2. Build the examples by creating a build directory, running CMake, and compiling the source files:

```sh
mkdir build
cd build
cmake ../
make
```

3. Once the compilation is complete, you can run each example by executing the corresponding executable file. For example, to run the object detection example:

```sh
./cpp/example_object_detection
```

Feel free to explore the different examples in the repository and modify them to suit your specific needs. Each example demonstrates the usage of a specific computer vision task, along with making API requests to the BrainyPi AI REST server.

## Using the OpenAPI Description

If you want to write your own code using the BrainyPi AI REST server API, you can utilize the OpenAPI description provided in the [openapi.yaml](openapi.yaml) file. The OpenAPI description defines the available endpoints, request and response structures, and the supported operations.

To use the OpenAPI description for writing custom code, follow these steps:

1.	Open the [openapi.yaml](openapi.yaml) file in a text editor or an OpenAPI-compatible tool.

2.	Familiarize yourself with the API endpoints, request parameters, and response structures. This information will help you understand how to construct requests and process the server's responses.

3.	Use an OpenAPI code generator or an HTTP client library compatible with OpenAPI specifications to generate code bindings or manually construct the HTTP requests.

4.	Incorporate the generated or manually written code into your project and customize it as needed.

By leveraging the OpenAPI description, you can ensure compatibility and consistency while interacting with the BrainyPi AI REST server API.

## License

This repository is licensed under the MIT License. Please refer to the [LICENSE](LICENSE) file for more information.

## Support

If you encounter any issues or have any questions regarding the BrainyPi AI API Examples, please open an issue in this repository. We'll do our best to assist you.

Happy coding!

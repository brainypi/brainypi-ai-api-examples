# Images example
add_executable(example_object_detection example_object_detection.cpp helper.cpp)
target_link_libraries(example_object_detection PRIVATE ${OpenCV_LIBS} )

# Images example
add_executable(example_face_detection example_face_detection.cpp helper.cpp)
target_link_libraries(example_face_detection PRIVATE ${OpenCV_LIBS} )

# Images example
add_executable(example_image_classification example_image_classification.cpp helper.cpp)
target_link_libraries(example_image_classification PRIVATE ${OpenCV_LIBS} )

# Images example
add_executable(example_pose_detection example_pose_detection.cpp helper.cpp)
target_link_libraries(example_pose_detection PRIVATE ${OpenCV_LIBS} )

# Images example
add_executable(example_face_registration example_face_registration.cpp helper.cpp)
target_link_libraries(example_face_registration PRIVATE ${OpenCV_LIBS} )

# Images example
add_executable(example_face_verification example_face_verification.cpp helper.cpp)
target_link_libraries(example_face_verification PRIVATE ${OpenCV_LIBS} )


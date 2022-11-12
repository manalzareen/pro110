# To Capture Frame
import cv2
import tensorflow as tf
# To process image array
import numpy as np


# import the tensorflow modules and load the model
models=tf.keras.models.load_model("keras_model.h5")


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		# get predictions from the model
		prediction=model.predict(frame)
		print("Prediction: ",prediction)
		
		#resize the frame
		img=cv2.resize(frame(244,244))
		
		# expand the dimensions
		testing_img = np.array(img,dtype=np.float32)
		testing_img=np.expand_dims(testing_img,axis = 0)
		# normalize it before feeding to the model
		normalized_img = testing_img/255.0

		prediction=predict(normalized_img)
		
		print("prediction: ",prediction)
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("."))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
cnt = 1
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	img_name = imagePath.split(os.path.sep)[-1]
	print(img_name)
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="hog")

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
            
	# Create a directory to save the face images
	output_dir = "detected_faces"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Loop over the face detections and save each face
	for i, (top, right, bottom, left) in enumerate(boxes):
		# Extract the face from the image
		face = image[top:bottom, left:right]

		# Construct the path to save the face image
		face_filename = os.path.join(output_dir, f"faces_{cnt}.jpg")
		cnt+=1
		# Save the face image
		cv2.imwrite(face_filename, face)

		print(f"Face {i+1} saved at {face_filename}")
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

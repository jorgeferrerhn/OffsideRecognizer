import numpy as np
import cv2
import os
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import math

NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2


def get_biggest_lines(lines, number=2):
    number = 2 if number < 1 else None
    
    ordered = sorted(lines, key= lambda l : l[0]) #order the lines with respect to their size
    #print("ordered: ",str(ordered))
    if (len(ordered) >= number):
        return ordered
    return ordered[0:number]


def get_vertical_lines(lines): #Angles should be expresed in radians
	return get_lines_oriented(lines, 0.61, 2.53073)


def get_lines_oriented(lines, min_angle, max_angle): #Angles should be expresed in radians
	return [l for l in lines if min_angle <= l[1] <= max_angle]


def draw_lines_simple(lines,image):

	print(len(lines))
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

			l = cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

def draw_lines_show_image(auxiliar_lines, attacker_line, defender_line, decision, image):
    red = (5,5,255)
    green = (20, 255, 110)
    blue = (255, 100, 15)
    
    #Draw auxiliar lines
    for line in auxiliar_lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
    #Draw defender line
    rho = defender_line[0][0]
    theta = defender_line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image, pt1, pt2, blue, 3, cv2.LINE_AA)
    
    #Draw attacker line
    rho = defender_line[0][0]
    theta = defender_line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    if decision:
        cv2.line(image, pt1, pt2, red, 3, cv2.LINE_AA) # red because is offsede
    else:
        cv2.line(image, pt1, pt2, green, 3, cv2.LINE_AA) # green because is not offside
     
    cv2.imshow(image, 0)
    
    return None



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results


def rectangle_mask(shape, vertex1, vertex2):
    rectangle = np.zeros(shape[:2], dtype="uint8")
    cv2.rectangle(rectangle, vertex1, vertex2, 255, -1)
    return rectangle


#------------------------------------------------------------------
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#------------------------------------------------------------------


image = cv2.imread("datasets/test/images/test3.jpg")
greyImage = cv2.imread("datasets/test/images/test3.jpg",cv2.IMREAD_GRAYSCALE)
dst = cv2.Canny(greyImage, 50, 200, None, 3)
plt.imshow(dst)
plt.show()

#Detect the lines
lines = cv2.HoughLines(greyImage, 1, math.pi / 180, 50, None, 0, 0)
layer_name = model.getLayerNames()
layer_name = [layer_name[i-1] for i in model.getUnconnectedOutLayers()]
results = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))
mask = np.zeros(image.shape[:2], dtype="uint8")

#Get the two biggest vertical lines

print("Len: ",len(lines.tolist()))
#auxiliar_lines = get_biggest_lines(get_vertical_lines(lines.tolist()))

draw_lines_simple(lines,dst)

#plt.imshow(image)
#plt.show()

plt.imshow(dst)
plt.show()


'''
for res in results:
	rectangle = rectangle_mask(image.shape, (res[1][0],res[1][1]), (res[1][2],res[1][3]))	
	mask = cv2.bitwise_or(mask, rectangle)
masked = cv2.bitwise_and(image, image, mask=mask)

plt.imshow(masked)
plt.show()


cap = cv2.VideoCapture("real_madrid_trim.mp4")
writer = None



while True:
	(grabbed, image) = cap.read()

	if not grabbed:
		break
	image = imutils.resize(image, width=700)
	results = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))

	for res in results:
		cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

	cv2.imshow("Detection",image)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()

'''

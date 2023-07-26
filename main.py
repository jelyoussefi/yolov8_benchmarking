import os, argparse
from pathlib import Path
from typing import Tuple, Dict
import torch
import cv2
import numpy as np
from ultralytics.yolo.utils.plotting import colors
from openvino.runtime import Core, Model
from ultralytics import YOLO
from time import perf_counter
from collections import deque
import psutil
import pathlib

def draw_perf(image, device, fps, infer_fps, cpu_load):
	frame_size = image.shape[:-1]
	fontFace =  cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.4
	thickness = 1
	margin = 15
	bcolor = (0,255,0)
	fcolor = (0,0,255)
	
	def circle(text, radius, pos, left=True, bcolor = (0,255,0), fcolor = (0,0,255), legend=""):
		textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

		if left:
			x = margin + 2*(radius+5)*pos + radius/2
		else :
			x = frame_size[1] - margin - 2*(radius+5)*pos - radius/2

		center = (int(x), int(margin + radius / 2))
		cv2.circle(image, center, radius, bcolor, 1, cv2.LINE_AA)
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, text, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)

		textsize = cv2.getTextSize(legend, fontFace, fontScale, thickness)[0]
		center = (int(x), int(margin + radius*2))
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, legend, textPos, fontFace, 0.4, (255,255,255), thickness, cv2.LINE_AA)

	# device name & infer fps
	if device == "MYRIAD":
		device = "VPU"
	infer_fps = f"{int(infer_fps)}"
	circle(device, 18, 0)
	circle(infer_fps, 18, 1, legend="inf. fps", fcolor=(255,0,0))

	# fps
	fps = f"{int(fps)}"
	circle(fps, 18, 1, False, legend="fps")

	#cpu load
	cpu_load = f"{int(cpu_load)}"
	circle(cpu_load, 18, 0, False, legend="%cpu", fcolor=(255,0,0))

	return image

class YoloV8Model():
	def __init__(self, model_path, device):
		self.device = device
		
		self.model = YOLO(model_path)
		self.label_map = self.model.model.names
		self.model.export(format="openvino", dynamic=True, half=True)
		model_dirname = os.path.dirname(model_path)
		model_name = os.path.basename(model_path).split('.')[0]
		model_path = os.path.join(model_dirname, model_name+"_openvino_model", model_name+".xml")
	
		self.core = Core()
		self.ov_model = self.core.read_model(model_path)
		if device != "CPU":
			self.ov_model.reshape({0: [1, 3, 640, 640]})

		self.compiled_model = self.core.compile_model(self.ov_model, device)

		input_layer = self.ov_model.input(0)
		self.input_shape = (input_layer.shape[2], input_layer.shape[3])

	def detect(self, image:np.ndarray, conf_threshold=0.5):
		
		image_height = image.shape[0]
		image_width = image.shape[1]

		preprocessed_image = self.preprocess(image)
		start_time = perf_counter()
		result = self.compiled_model(preprocessed_image)
		infer_time = perf_counter() - start_time
		outputs = result[self.compiled_model.output(0)]
		outputs = np.array([cv2.transpose(outputs[0])])
		rows = outputs.shape[1]

		boxes = []
		scores = []
		class_ids = []

		for i in range(rows):
			classes_scores = outputs[0][i][4:]
			(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
			if maxScore >= conf_threshold:
				box = [
				    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
				    outputs[0][i][2], outputs[0][i][3]]
				boxes.append(box)
				scores.append(maxScore)
				class_ids.append(maxClassIndex)

		indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.7, 0.5)
		if len(indexes) > 0:
			boxes = self.rescale_boxes(boxes, (image_height, image_width))

			for i in range(len(indexes)):
				index = indexes[i]
				box = boxes[index]
				self.plot_one_box(image, class_ids[index], scores[index], box[0], box[1], box[0] + box[2], box[1] + box[3])

		return image, infer_time


	def preprocess(self, input_image):
		image = cv2.resize(input_image, (self.input_shape[1], self.input_shape[0]))
		image = image.transpose(2, 0, 1) # Convert HWC to CHW
		image = image.astype(np.float32)/255.0
		if image.ndim == 3:
			image = np.expand_dims(image, 0)
		return image

	def plot_one_box(self, image, class_id, confidence, x1, y1, x2, y2, line_thickness:int = 2):
		color = colors(class_id)
		tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
		c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
		cv2.rectangle(image, c1, c2, color=color, thickness=tl, lineType=cv2.LINE_AA)
		
		tf = max(tl - 1, 1)  # font thickness
		label = f'{self.label_map[class_id]} {confidence:.2f}'
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


	def rescale_boxes(self, boxes, image_shape):
		# Rescale boxes to original image dimensions
		input_shape = np.array([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]])
		boxes = np.divide(boxes, input_shape, dtype=np.float32)
		boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

		return boxes

	
def run(model_path, device, input):
	
	model = YoloV8Model(model_path, device)

	cap = cv2.VideoCapture(input)
	
	cpu_loads = deque(maxlen=50)
	frames_number = 0
	infer_times = []
	cpu_loads.append(psutil.cpu_percent(0.1))

	start_time = perf_counter()

	while True:
		st = perf_counter()
		ret, image = cap.read()

		if ret is False:
			if frames_number == 0:
				break
			else:
				cap = cv2.VideoCapture(input)
				continue

		frames_number += 1
		
		image, infer_time = model.detect(image)
		infer_times.append(infer_time)
		cpu_loads.append(psutil.cpu_percent(0))

		cpu_load = np.average(cpu_loads);
		infer_fps = 1/np.average(infer_times);
		fps = frames_number/(perf_counter() - start_time)

		image = draw_perf(image, device, fps, infer_fps, cpu_load)
		cv2.imshow("yolov8 demo", image)

		key = cv2.waitKey(1)
		if key in {ord('q'), ord('Q'), 27}:
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default=None, help='model path')
	parser.add_argument('--device', default="CPU", help='device type CPU or GPU')
	parser.add_argument('--input', default=None, help='input file')

	args = parser.parse_args()
		
	run(args.model, args.device, args.input)

	

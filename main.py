"""
export PYTHONPATH=/Users/gene/Code/Python/caffe/python
export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib:/opt/intel/composer_xe_2015.3.187/compiler/lib:/opt/intel/composer_xe_2015.3.187/mkl/lib
"""

import deepdream as dd
import math
	
def animation():
	imagePath = '/Users/gene/Documents/Processing/noiseField/frames/noiseframeFull2.jpg'
	#imagePath = '/Users/gene/Documents/Processing/noiseField/frames/noiseframeFull2Small.jpeg'
	classifier_index = 1
	name = 'testSeq1Large2/test'
	end1 = 'inception_4c/output'
	end2 = 'inception_3b/5x5_reduce'
	generations = 720
	iterations = 10
	octaves = 4
	octave_scale = 1.4
	jitter = 16
	scaleZoom = 0.02
	
	img = dd.loadImage(imagePath)
	classifier = dd.setupModel(classifier_index)
	frame = img
	
	for gen in range(generations):
		iterations = int(7 + 6*math.sin(gen*0.1))
		octaves = int(5+4*math.sin(gen*0.2+5))
		octave_scale = round(1.2 + 0.2*math.sin(gen*0.3-7), 1)
		scaleZoom = round(0.03 + 0.02*math.sin(gen*0.02+19), 2)
		
		if   (gen < 44):	end = end2
		elif (gen < 64):	end = end1
		elif (gen < 184):	end = end2
		elif (gen < 304):	end = end1
		elif (gen < 380):	end = end2
		elif (gen < 445):	end = end1
		elif (gen < 495):	end = end2
		elif (gen < 535):	end = end1
		elif (gen < 565):	end = end2
		elif (gen < 595):	end = end1
		elif (gen < 615):	end = end2
		elif (gen < 635):	end = end1
		elif (gen < 645):	end = end2
		elif (gen < 655):	end = end1
		elif (gen < 665):	end = end2
		elif (gen < 675):	end = end1
		elif (gen < 682):	end = end2
		elif (gen < 689):	end = end1
		elif (gen < 694):	end = end2
		elif (gen < 710):	end = end1
		else:	end = end2
		#if int(gen / 44) % 2 == 0:
		#	end = end2
		#else:
		#	end = end1
	
		print "G "+str(gen)+"/"+str(generations)+" ("+end+") iter "+str(iterations)+", octaves "+str(octaves)+", oscale "+str(octave_scale)+", jitter "+str(jitter)+", scaleZoom "+str(scaleZoom)
		frame = dd.makeDeepDream(frame, classifier, end, name+'%04d'%gen, iterations, octaves, octave_scale, jitter, scaleZoom)	

def run_trainingSet():
	img = dd.loadImage('/Users/gene/Code/Python/deepdream/pictures/trainingSet.jpg')
	classifier = dd.setupModel(1)
	name = 'trainingSetOut'
	end1 = 'inception_4c/output'
	end2 = 'inception_3b/5x5_reduce'
	end = end1
	generations = 10
	iterations = 10
	octaves = 4
	octave_scale = 1.4
	jitter = 32
	scaleZoom = 0.00
	
	frame = img
	for gen in range(generations):
		print str(gen)+"/"+str(generations)
		frame = dd.makeDeepDream(img, classifier, end, name+'%04d'%gen, iterations, octaves, octave_scale, jitter, scaleZoom)	
	
def animation_stepped():
	imagePath = '/Users/gene/Documents/Processing/noiseField/frames/noiseframeFull2.jpg'
	imagePath = '/Users/gene/Documents/Processing/noiseField/frames/noiseframeFull2Small.jpeg'
	classifier_index = 1
	name = 'testSeq1stepped4/test'
	end1 = 'inception_4c/output'
	end2 = 'inception_3b/5x5_reduce'
	generations = 720
	iterations = 10
	octaves = 4
	octave_scale = 1.4

	start_jitter=48.
	end_jitter=4.
	start_step_size=3.0
	end_step_size=1.5
	start_sigma=2.5
	end_sigma=.1

	scaleZoom = 0.05

	img = dd.loadImage(imagePath)
	classifier = dd.setupModel(classifier_index)
	frame = img

	for gen in range(generations):
		iterations = int(9 + 6*math.sin(gen*0.1))
		octaves = int(5+4*math.sin(gen*0.2+5))
		octave_scale = round(1.2 + 0.2*math.sin(gen*0.3-7), 1)
		scaleZoom = round(0.03 + 0.02*math.sin(gen*0.02+19), 2)

		start_jitter = int(30 + 18*math.sin(gen*0.1-98))
		end_jitter = int(6 + 4*math.sin(gen*0.1+71))
		start_step_size = round(2.5 + 1.2*math.sin(gen*0.01+119), 2)
		end_step_size = round(0.9 + 0.3*math.sin(gen*0.012+219), 2)
		start_sigma = round(1.0 + 0.5*math.sin(gen*0.015+319), 2)
		end_sigma = round(0.3 + 0.2*math.sin(gen*0.009+419), 2)

		if int(gen / 44) % 2 == 0:
			end = end2
		else:
			end = end1

		print "G "+str(gen)+"/"+str(generations)+" ("+end+") iter "+str(iterations)+", octaves "+str(octaves)+", oscale "+str(octave_scale)+", jitter "+str(start_jitter)+"-"+str(end_jitter)+", scaleZoom "+str(scaleZoom)
		frame = dd.makeDeepDreamStepped(frame, classifier, end, name+'%04d'%gen, iterations, octaves, octave_scale, start_sigma, end_sigma, start_jitter, end_jitter, start_step_size, end_step_size, scaleZoom)	


dd.setPathToCaffe('/Users/gene/Code/Python/caffe/')
dd.setOutputDirectory('/Users/gene/Desktop/output/')

animation_stepped()
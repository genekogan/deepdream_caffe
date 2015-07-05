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
	name = 'testSeq1Large/test'
	end1 = 'inception_4c/output'
	end2 = 'inception_3b/5x5_reduce'
	generations = 720
	iterations = 10
	octaves = 4
	octave_scale = 1.4
	jitter = 16
	scaleZoom = 0.05
	
	img = dd.loadImage(imagePath)
	classifier = dd.setupModel(classifier_index)
	frame = img
	
	for gen in range(generations):
		iterations = int(7 + 6*math.sin(gen*0.1))
		octaves = int(5+4*math.sin(gen*0.2+5))
		octave_scale = round(1.2 + 0.2*math.sin(gen*0.3-7), 1)
		scaleZoom = round(0.03 + 0.02*math.sin(gen*0.02+19), 2)
		
		if int(gen / 44) % 2 == 0:
			end = end2
		else:
			end = end1
	
		print "G "+str(gen)+"/"+str(generations)+" ("+end+") iter "+str(iterations)+", octaves "+str(octaves)+", oscale "+str(octave_scale)+", jitter "+str(jitter)+", scaleZoom "+str(scaleZoom)
		frame = dd.makeDeepDream(frame, classifier, end, name+'%04d'%gen, iterations, octaves, octave_scale, jitter, scaleZoom)	


dd.setPathToCaffe('/Users/gene/Code/Python/caffe/')
dd.setOutputDirectory('/Users/gene/Desktop/output/')

animation()
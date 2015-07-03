import deepdream as dd


myDirectory = '/Users/gene/Code/Python/deepdream/pictures/'

dd.setPathToCaffe('/Users/gene/Code/Python/caffe/')
dd.setOutputDirectory('/Users/gene/Desktop/output/')

classifier = dd.setupModel(1)
img = dd.loadImage(myDirectory+'alec1.jpg')
prefix = 'alec/alec'

generations = 1
for iterations in range(5,28,7):
	for octaves in range(1,11,3):
		for octave_scale in [0.7, 2.1]:
			print "generate " + str(iterations) + ", " + str(octaves) + ", " + str(octave_scale)
			dd.makeDeepDream(img, classifier, prefix, generations, iterations, octaves, octave_scale, 0.05)
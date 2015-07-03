working with google's deep dream codebase. 

to install, follow the instructions [here](https://github.com/google/deepdream/blob/master/dream.ipynb). instructions for installing [caffe are here (not straightforward)](http://caffe.berkeleyvision.org/installation.html). if you are having trouble installing caffe on OSX, [this writeup helped me](http://hoondy.com/2015/04/03/how-to-install-caffe-on-mac-os-x-10-10-for-dummies-like-me/).


### Usage

after installing, make sure you add pycaffe to your path, e.g.

    export PYTHONPATH=/Users/gene/Code/Python/caffe/python

if you are (like me) using anaconda, you may need to also add these to your path (as described by the last link), e.g.

    export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib:/opt/intel/composer_xe_2015.3.187/compiler/lib:/opt/intel/composer_xe_2015.3.187/mkl/lib



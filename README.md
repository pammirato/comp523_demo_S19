
Example object detector with pytorch



I recommend using `virtual_env` to install all the python stuff, maybe

Then run 

`pip install -r requirements.txt`

to install the two python dependencies,
then install pytorch https://pytorch.org/get-started/locally/

Then download the model parameters: https://drive.google.com/file/d/1eXgdNp0YlcCXVLyljOrQmyUi0AB3GWKF/view?usp=sharing

It should be fine if you don't have a GPU and/or CUDA, this example should run fine on a cpu

`python pytorch_example.py`




**NOTE:** I've been saying the images should be numpy arrays, which is good. We could also use PIL images instead if that is easier for you all. Or I can convert the numpy image to PIL later if I need to. So at the least we should make sure we can have the python PIL library imported






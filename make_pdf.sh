#!/bin/sh

jupyter nbconvert --to=pdf --template=./templates/article.tplx --output-dir=./pdf/ README.ipynb

cd 10_python

jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 11_Introduction.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 12_pip.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 13_Anaconda.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 14_Jupyter_Notebook.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 15_A_Byte_of_Python.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 16_Numpy.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 17_Matplotlib.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 18_Scikit_learn.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/10_python/ 19_Linear_Algebra.ipynb

cd ../20_opencv

jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 21_Introduction.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 22_Image_Processing.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 23_Drawing_Functions.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 24_Feature_Detection.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 25_Video_Processing.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 26_Machine_Learning.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/20_opencv/ 27_Object_Detection.ipynb

cd ../30_deep_learning

jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 31_Bayesian_Inference.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 32_Linear_Classification.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 33_MLP.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 34_CNN.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 35_Supervised_Learning.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/30_deep_learning/ 36_Applications.ipynb

cd 37_Yolo

jupyter nbconvert --to=pdf --template=../../templates/article.tplx --output-dir=../../pdf/30_deep_learning/ 37_Yolo.ipynb
cd ..

cd ../40_tensorflow

jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/40_tensorflow/ 41_Introduction.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/40_tensorflow/ 42_Basics.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/40_tensorflow/ 43_Keras.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/40_tensorflow/ 44_Models.ipynb
jupyter nbconvert --to=pdf --template=../templates/article.tplx --output-dir=../pdf/40_tensorflow/ 45_TensorBoard.ipynb

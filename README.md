# nybg-resnet

The following has been taken from NYBG's problem statement:

Researchers at NYBG and across the world have access to millions of digitized images. These images – which include pressed and dried plant specimens – are searchable in online databases (such as the Global Biodiversity Information Facility). They can help the researchers conduct botanical studies spanning comparative biodiversity, applied conservation, and climate modeling: studies important to the future of planet Earth.

The problem is, approximately 10% of the 40+ million database images are “non-standard.” For example, they might be images of animals instead of plants; or color illustrations of plants rather than actual pressed specimens. These non-standard images generally cannot be used as part of cutting-edge studies that aim to leverage machine learning to do things like train models to recognize new plant species, or predict and analyze biodiversity-related changes over time. There’s no way for the researchers to easily tell which images from their database search are non-standard (and might therefore need to be excluded from their machine learning dataset curation).

![image](https://github.com/user-attachments/assets/b9995f94-e4ea-4a63-803c-f5a05f4c02a7)

### Solving the problem
The final model was a pretrained ResNet101 model using SGD as an optimizer with a learning rate of 0.01, yielding an accuracy of 0.9827.

The second best model was a ResNet50 model, which maxed out at an accuracy of 0.97. 

### Preprocessing
Preprocessing involved several features including scaling and rotating to allow for more robust image input into the model. These values were adjusted several times but it was found that the current values in the code led to the best accuracy. 

### Model Architecture
GlobalAveragePooling2D() and numerous dense layers with varying input shapes of 256 - 1024 were added, however this would often lead to overfitting.

### Fine Tuning Hyperparameters
With the architecture determined the models hyperparameters were adjusted. Various optimizers were tried including RMSProp and SGD, which yielded the best results. Additionally various learning rates from 0.000001 to 0.01 were tried. 

ResNet50 could not reach past a 97% accuracy, but ResNet101 exceded this threshold.

Each epoch took around 30 minutes to train and 8 epochs tended to be the best in terms of accuracy.

Moving forward further adjusting the hyperparameters, such as adding momentum to the optimizer and even using ensemble methods could lead to a more robust model.

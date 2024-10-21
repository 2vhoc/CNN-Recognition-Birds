import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.src.utils import to_categorical
Xtrain = []
yTrain = []
shape = ((224, 224), 3)
path = 'datasetsBird'
for name in os.listdir(path):
    namePath = os.path.join(path, name)
    nameSplit = namePath.split('\\')[1]
    yTrain.append(nameSplit)
i = 0
for folder in os.listdir(path):
    typeBird = []
    folderPath = os.path.join(path, folder)
    for img in os.listdir(folderPath):
        imgPath = os.path.join(folderPath, img)
        imgNp = np.array(Image.open(imgPath))
        nameBird = i
        nameBird = to_categorical(nameBird, num_classes=len(yTrain))
        typeBird.append((imgNp, nameBird))
    Xtrain.extend(typeBird)
    i += 1
from tensorflow.keras.models import load_model
H = load_model('modelBird.h5')
a = img = cv2.imread(r'D:\Recognition Bird CNN\datasetsBird\FLAMINGO\040.jpg')
img = cv2.resize(img, (224, 224)).reshape((-1, 224, 224, 3))
img = np.array(img).astype('float')/255.0
result = H.predict(img)
result = np.argmax(result, axis=1)
cv2.putText(a, f'{yTrain[result[0]]}', (1, 15) ,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)


print(result)
print(yTrain[result[0]])
print('ok')
plt.imshow(a)
plt.show()

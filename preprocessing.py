import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

def len_data(filename):
    anger_data = os.listdir(filename+'/anger')
    happy_data = os.listdir(filename+'/happy')
    sadness_data = os.listdir(filename+'/sadness')
    surprise_data = os.listdir(filename+'/surprise')

    value = []
    for data in [anger_data,happy_data,sadness_data,surprise_data]:
        value.append(len(data))
        
    return sum(value)

filename = 'CK+48'
print('Total Images in set : ' + str(len_data(filename)))

import cv2

def load_images_from_folder(folder):
    images = []
    
    folder1 = folder + '\\anger'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(),folder1,filename))
        if img is not None:
            images.append(img)
            
    folder4 = folder + '\\happy'
    for filename in os.listdir(folder4)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(),folder4,filename))
        if img is not None:
            images.append(img)
            
    folder5 = folder + '\\sadness'
    for filename in os.listdir(folder5)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(),folder5,filename))
        if img is not None:
            images.append(img)
    
    folder6 = folder + '\\surprise'
    for filename in os.listdir(folder6)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(),folder6,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder('CK+48')

## Plotting the count of each emotion....
fig, axis = plt.subplots(4,20,figsize=(20,7))
count = 0
for i in range(0,4):
    for j in range(0,20):
        img_rgb = cv2.cvtColor(images[count], cv2.COLOR_BGR2RGB)
        axis[i,j].imshow(img_rgb)
        axis[i,j].axis('off')
        count = count + 1
        if i==0:
            axis[i,j].set_title('anger')
        elif i==1:
            axis[i,j].set_title('happy')
        elif i==2:
            axis[i,j].set_title('sad')
        else:
            axis[i,j].set_title('surprise')

## plotting sample faces...
def plot_data(filename):
    buildings_data = os.listdir(filename+'/anger')
    mountain_data = os.listdir(filename+'/happy')
    sea_data = os.listdir(filename+'/sadness')
    street_data = os.listdir(filename+'/surprise')

    value = []
    for data in [buildings_data,mountain_data,sea_data,street_data]:
        value.append(len(data))
    
    sns.barplot(['angry','happy','sad','surprise'],value, palette = 'plasma')

filename = 'CK+48'
plot_data(filename)
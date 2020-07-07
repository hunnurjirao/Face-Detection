import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
 
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path,1)
    img1 = cv2.resize(img1,(256,256))
    img1 = np.around(img1/255.0)
    x_train = np.array([img1])
    encoded = model.predict_on_batch(x_train)
    return encoded

def predict(image_path, id, database, model):

    encoding = img_to_encoding(image_path, model)
    
    dist = np.linalg.norm(encoding - database[id])
    print(dist)

    if dist <0.5:
        res = ("It's " + str(id) + ", welcome!")
    else:
        res = ("It's not " + str(id) + ", please go away")

    img=mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(res)
    plt.show()
    return dist
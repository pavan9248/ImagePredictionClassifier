{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "fdafec57",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras import datasets , layers , models\n",
    "import matplotlib.pyplot as plt \n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "e432683f",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "d778c9b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(50000, 32, 32, 3)"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "532079fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 32, 32, 3)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "291f7b59",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6, 9, 9, 4, 1], dtype=uint8)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[:5]\n",
    "y_train=y_train.reshape(-1,)\n",
    "y_train[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "39b4daa2",
   "metadata": {},
   "outputs": [],
   "source": [
    "classes=[\"airplane\",\"automobile\",\"bird\",\"cat\",\"deer\",\"dog\",\"frog\",\"horse\",\"ship\",\"truck\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "2da11198",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_sample(x,y,index):\n",
    "    plt.figure(figsize=(15,2))\n",
    "    plt.imshow(x_train[index])\n",
    "    plt.xlabel(classes[y[index]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "91ec81a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x144 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_sample(x_train,y_train,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "4255ab17",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train=x_train/255\n",
    "x_test=x_test/255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "bc64bf56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "1563/1563 [==============================] - 156s 99ms/step - loss: 1.8842 - accuracy: 0.3276\n",
      "Epoch 2/5\n",
      "1563/1563 [==============================] - 154s 99ms/step - loss: 1.6731 - accuracy: 0.3981\n",
      "Epoch 3/5\n",
      "1563/1563 [==============================] - 151s 97ms/step - loss: 1.5971 - accuracy: 0.4279\n",
      "Epoch 4/5\n",
      "1563/1563 [==============================] - 151s 96ms/step - loss: 1.5464 - accuracy: 0.4470\n",
      "Epoch 5/5\n",
      "1563/1563 [==============================] - 151s 96ms/step - loss: 1.5059 - accuracy: 0.4593\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x1a6000bed00>"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model=models.Sequential([\n",
    "    layers.Flatten(input_shape=(32,32,3)),\n",
    "    layers.Dense(3000,activation='relu'),\n",
    "    layers.Dense(1000,activation='relu'),\n",
    "    layers.Dense(10,activation='sigmoid')\n",
    "])\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "model.fit(x_train,y_train,epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "d6106f66",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 6s 19ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.59      0.48      0.53      1000\n",
      "           1       0.57      0.65      0.61      1000\n",
      "           2       0.33      0.30      0.32      1000\n",
      "           3       0.30      0.39      0.33      1000\n",
      "           4       0.45      0.31      0.37      1000\n",
      "           5       0.41      0.36      0.39      1000\n",
      "           6       0.43      0.66      0.52      1000\n",
      "           7       0.49      0.54      0.51      1000\n",
      "           8       0.70      0.51      0.59      1000\n",
      "           9       0.57      0.50      0.53      1000\n",
      "\n",
      "    accuracy                           0.47     10000\n",
      "   macro avg       0.48      0.47      0.47     10000\n",
      "weighted avg       0.48      0.47      0.47     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix,classification_report\n",
    "ypred=model.predict(x_test)\n",
    "ypred_class=[np.argmax(element) for element in ypred]\n",
    "print(classification_report(y_test,ypred_class))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "990bf86a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def pre():\n",
    "    cnn=models.Sequential([\n",
    "        layers.Conv2D(filters=32,input_shape=(32,32,3),kernel_size=(3,3),activation='relu'),\n",
    "        layers.MaxPooling2D((2,2)),\n",
    "\n",
    "        layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),\n",
    "        layers.MaxPooling2D((2,2)),\n",
    "\n",
    "\n",
    "        layers.Flatten(),\n",
    "        layers.Dense(3000,activation='relu'),\n",
    "        layers.Dense(1000,activation='relu'),\n",
    "        layers.Dense(10,activation='sigmoid')\n",
    "    ])\n",
    "    cnn.compile(optimizer='adam',\n",
    "                  loss='sparse_categorical_crossentropy',\n",
    "                  metrics=['accuracy'])\n",
    "    cnn.fit(x_train,y_train,epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "e2e0d375",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 9s 29ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.72      0.81      0.76      1000\n",
      "           1       0.81      0.86      0.83      1000\n",
      "           2       0.75      0.48      0.59      1000\n",
      "           3       0.47      0.61      0.53      1000\n",
      "           4       0.63      0.68      0.65      1000\n",
      "           5       0.64      0.58      0.61      1000\n",
      "           6       0.79      0.77      0.78      1000\n",
      "           7       0.78      0.76      0.77      1000\n",
      "           8       0.83      0.79      0.81      1000\n",
      "           9       0.80      0.78      0.79      1000\n",
      "\n",
      "    accuracy                           0.71     10000\n",
      "   macro avg       0.72      0.71      0.71     10000\n",
      "weighted avg       0.72      0.71      0.71     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "ypred1=cnn.predict(x_test)\n",
    "ypred1_class=[np.argmax(element) for element in ypred1]\n",
    "print(classification_report(y_test,ypred1_class))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "13ffb22b",
   "metadata": {},
   "outputs": [],
   "source": [
    "tf.keras.models.save_model(cnn,'my_model.hdf5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bea162f4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

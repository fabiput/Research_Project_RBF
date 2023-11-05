# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from keras                   import backend as K
from keras.layers            import Layer
from keras.initializers      import RandomUniform, Initializer, Constant
from keras.initializers      import Initializer
from sklearn.cluster         import KMeans, AffinityPropagation, MeanShift, DBSCAN
from keras.models            import Sequential 
from keras.layers            import Dense
from keras.layers            import Activation
from keras.optimizers        import RMSprop
from sklearn.decomposition   import PCA
from sklearn.preprocessing   import OneHotEncoder
from sklearn.metrics         import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# class for initialization of random cluster centers (given as random samples from the dataset)
class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0) #initialization of basis functions
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name        = 'centers',
                                       shape       = (self.output_dim, input_shape[1]),
                                       initializer = self.initializer,
                                       trainable   = True)
        self.betas   = self.add_weight(name        = 'betas',
                                       shape       = (self.output_dim,),
                                       initializer = Constant(value=self.init_betas),
                                       trainable   = True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        print('call return',K.exp(-self.betas * K.sum(H**2, axis=1)))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        n_centers = shape[0]
        print('These are centers: ',n_centers)
        #KM
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        #AF
        af = AffinityPropagation()
        af.fit_predict(X)
        #MS
        ms= MeanShift()
        ms.fit_predict(X)
        print('cluster centers: ',af.cluster_centers_)
        return af.cluster_centers_

        # return km.cluster_centers_
    
    # class InitCentersAF(Initializer):
    #     """ Initializer for initialization of centers of RBF network
    #     by clustering the given data set.
    #     # Arguments
    #     X: matrix, dataset
    #     """

    # def __init__(self, X, max_iter=100):
    #     self.X = X
    #     self.max_iter = max_iter

    # def __call__(self, shape, dtype=None):
    #     assert shape[1] == self.X.shape[1]
    #     af = AffinityPropagation()
    #     af.fit_predict(X)
    #     print('These are centers: ',len(af.cluster_centers_))
    #     return af.cluster_centers_


#data Loading
# data = pd.read_csv('olive.csv',header=None)
# datatrans=np.transpose(data)

# #af and pca initialization
pca = PCA(n_components=2)

# # #data spliting
# X = data.iloc[2:570,:].values
# y = data.iloc[0:1,:].values

# #data rotation
# X=np.transpose(X)
# y=np.transpose(y)

# iris dataset
from sklearn.datasets import load_iris

data =  load_iris()
X = data.data
y = data.target
y=y.reshape(y.shape[0],1)

X = pca.fit_transform(X)
print(X)
print(y)

#standarizing
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)

#one hot encoding targets
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# sc_y = StandardScaler()
# y_train = y_train.reshape((len(y_train), 1))
# y_train = sc_y.fit_transform(y_train)
# y_train = y_train.ravel()

#model building
model = Sequential()
rbflayer = RBFLayer(9, initializer=InitCentersKMeans(X_train), betas=3.0, input_shape=(X.shape[1],))
model.add(rbflayer)
model.add(Dense(3))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error',optimizer=RMSprop(), metrics=['accuracy'])
print(model.summary())
history1 = model.fit(X_train, y_train, epochs=2000, batch_size=64,verbose=0)

#train results
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('train accuracy and loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

#train accuracy
print(history1.history['accuracy'])

# saving to and loading from file
# z_model = f"Z_model.h5"
# print(f"Save model to file {z_model} ... ", end="")
# model.save(z_model)
# print("OK")

#model already saved in file
# from keras.saving import load_model    
# newmodel1= load_model("Zoghbio.h5",
#                           custom_objects={'RBFLayer': RBFLayer})
# print("OK")

#Evaluate the model on the test data using `evaluate`

#evaluation on test data
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss:", results[0])
print("test accuracy:",results[1]*100,'%')
y_pred = model.predict(X_test)

#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

layer_name = 'rbf_layer'  # Replace with the name of the desired layer
layer = model.get_layer(name=layer_name)
print(layer.get_weights()) 
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

#Results of test data
a    = accuracy_score(pred,test)
cfn  = confusion_matrix(test,pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cfn, display_labels= ['Class 0', 'Class 1','Class 2'])
disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' for integer formatting
plt.title("Confusion Matrix")
plt.show()
print('Test Accuracy is:', a*100)

print(test,y_test)
print(rbflayer.centers)


def plot_circle(center, radius,axis):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    axis.plot(x, y,c='m')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for k in range(rbflayer.centers.shape[0]):
    ax1.scatter(rbflayer.centers[k][0],rbflayer.centers[k][1],s=120, c='m', marker="o",zorder=2)
    plot_circle(rbflayer.centers[k],np.sqrt(2*rbflayer.betas[k])/5,ax1)

for i in range(X.shape[0]):
    if np.argmax(y[i]) == 0:
        ax1.scatter(X[i,0],X[i,1], s=10, c='r', marker="o", label='second')
    if np.argmax(y[i]) == 1:
        ax1.scatter(X[i,0],X[i,1], s=10, c='g', marker="o", label='second')
    if np.argmax(y[i]) == 2:
        ax1.scatter(X[i,0],X[i,1], s=10, c='y', marker="o", label='second')
ax1.set_aspect('equal', adjustable='box')

plt.show()


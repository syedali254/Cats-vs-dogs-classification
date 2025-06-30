import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import time

cat1 = io.imread("cat.1.jpg")
cat1_ = resize(cat1, (200,200,3))
cat1_gray = rgb2gray(cat1_)
fig = plt.figure()
columns = 3; rows = 1
fig.add_subplot(rows, columns, 1); plt.imshow(cat1)
fig.add_subplot(rows, columns, 2); plt.imshow(cat1_)
fig.add_subplot(rows, columns, 3); plt.imshow(cat1_gray)
plt.show()

x_train = []; y_train = []
for i in range(1,2001):
    cat = rgb2gray(resize(io.imread(f'training_set/cats/cat.{i}.jpg'), (200,200)))
    x_train.append(cat); y_train.append(0)
for i in range(1,2001):
    dog = rgb2gray(resize(io.imread(f'training_set/dogs/dog.{i}.jpg'), (200,200)))
    x_train.append(dog); y_train.append(1)
x_train, y_train = np.asarray(x_train), np.asarray(y_train)

x_test = []; y_test = []
for i in range(4001,5001):
    cat = rgb2gray(resize(io.imread(f'test_set/cats/cat.{i}.jpg'), (200,200)))
    x_test.append(cat); y_test.append(0)
for i in range(4001,5001):
    dog = rgb2gray(resize(io.imread(f'test_set/dogs/dog.{i}.jpg'), (200,200)))
    x_test.append(dog); y_test.append(1)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)

def predict(X, k):
    distances = [np.sum(np.abs(x_train[i] - X)) for i in range(len(x_train))]
    min_indexs = np.argsort(distances)[:k]
    y_ = y_train[min_indexs]
    counts = np.bincount(y_)
    return 'cat' if np.argmax(counts)==0 else 'dog'

numeros_images_a_predire = [4,1089]
fig = plt.figure(); predictions = []
columns = 2; rows = 1; i = 1
for num in numeros_images_a_predire:
    predictions += [predict(x_test[num], 3)]
    fig.add_subplot(rows, columns, i); plt.imshow(x_test[num]); i += 1
plt.show()
print(predictions)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

neigh = KNeighborsClassifier(n_jobs=-1)
neigh.fit(x_train, y_train)
print("Le score sur training set : {:.2f}%".format(neigh.score(x_train, y_train)*100))
print("Le score sur test set : {:.2f}%".format(neigh.score(x_test, y_test)*100))

distances = ['euclidean','cityblock']
valeurs_de_k = np.arange(1, 31, 2)
parametres_grid = {"n_neighbors": valeurs_de_k, 'metric': distances}

grid = RandomizedSearchCV(neigh, parametres_grid)
start = time.time()
grid.fit(x_train, y_train)
print("randomized search took {:.2f} minutes".format((time.time() - start)/60))
print("randomized search accuracy: {:.2f}%".format(grid.score(x_test, y_test) * 100))
print("randomized search best parameters: {}".format(grid.best_params_))

grid = GridSearchCV(neigh, parametres_grid)
start = time.time()
grid.fit(x_train, y_train)
print("grid search took {:.2f} minutes".format((time.time() - start)/60))
print("grid search accuracy: {:.2f}%".format(grid.score(x_test, y_test) * 100))
print("grid search best parameters: {}".format(grid.best_params_))

clf_gini = DecisionTreeClassifier(criterion = "gini")
clf_gini.fit(x_train, y_train)
print ("Accuracy_gini : {:.2f}%".format(clf_gini.score(x_test, y_test)*100))

clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(x_train, y_train)
print ("Accuracy_entropy : {:.2f}%".format(clf_entropy.score(x_test, y_test)*100))

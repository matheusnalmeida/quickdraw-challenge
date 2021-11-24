# Libs
from random import randint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
# Custom imports
import utils.util as util
from utils.classes import Classes
import utils.constants as constants
import matplotlib.pyplot as plt

# Carregando imagens
images_airplane = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_airplane.npy')
images_banana = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_banana.npy')
images_bee = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_bee.npy')
images_cofee_cup = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_coffee cup.npy')
images_crab = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_crab.npy')
images_guitar = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_guitar.npy')
images_hamburguer = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_hamburger.npy')
images_rabbit = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_rabbit.npy')
images_truck = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_truck.npy')
images_umbrella = util.load_quickdrow_images_from_npy('data/full_numpy_bitmap_umbrella.npy')

# Processamento dos dados
classes_dict = {}
sequencial = 0

for classe in Classes._member_names_:
    classificacao_atual = np.zeros(constants.NUMERO_DE_CLASSES)
    classificacao_atual[sequencial] = 1
    classes_dict[classe] = classificacao_atual
    sequencial += 1

# Montando variaveis x e y
x = []
y = []

util.generate_x_y_from_dataset(images_airplane, Classes.aviao.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_banana, Classes.banana.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_bee, Classes.abelha.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_cofee_cup, Classes.copo_de_cafe.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_crab, Classes.carangueijo.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_guitar, Classes.guitarra.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_hamburguer, Classes.hamburguer.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_rabbit, Classes.coelho.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_truck, Classes.caminhao.name, classes_dict, x, y)
util.generate_x_y_from_dataset(images_umbrella, Classes.guarda_chuva.name, classes_dict, x, y)

# Separando dados para treino e teste

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(x,
                                                                  y,
                                                                  test_size = 0.1,
                                                                  random_state = 0)

X_treinamento = np.array(X_treinamento)
X_teste = np.array(X_teste)
y_treinamento = np.array(y_treinamento)
y_teste = np.array(y_teste)

print("Quantidade de dados para treino: ", len(X_treinamento))

# Treinando a rede neural CNN(Convolutional Neural Network)

cnn_model = KerasClassifier(build_fn=util.create_cnn_model, epochs=50, verbose=0)

# Aplicando GridSearchCV para escolha dos melhores parametros

optimizer = ['Adam']

param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_treinamento, y_treinamento)

best_history = grid_result.best_estimator_.model.history.history 
plt.plot(best_history['loss'])
plt.plot(best_history['accuracy'])
plt.savefig('metrics/cnn/val_loss_acuracy_chart.png')

grid_result.best_estimator_.model.save('model/cnn_model.h5')

print("Melhor score: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))
medias_scores = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
parametros = grid_result.cv_results_['params']
for media_score, stdev, parametro in zip(medias_scores, stds, parametros):
    print("%f (%f) com: %r" % (media_score, stdev, parametro))

#util.generate_excel_metrics_crossvalidate(X_treinamento, y_treinamento, util.create_cnn_model, "cnn/cnn.xlsx")
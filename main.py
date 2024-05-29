from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib

#carregar a base de dados
wine_dataset = load_wine()
x = wine_dataset['data']
y = wine_dataset['target']
nome_das_classes = wine_dataset.target_names
descricao = wine_dataset['DESCR']
print(descricao)

#dividir o dataset em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=32)
print(f"Tamanho do conjunto de treino: {len(x_treino)}")
print(f"Tamanho do conjunto de teste: {len(x_teste)}")

#normalizar os dados
normalizador = MinMaxScaler()
normalizador.fit(x_treino)
x_treino_norm = normalizador.transform(x_treino)
x_teste_norm = normalizador.transform(x_teste)

#criar o classificador
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_treino_norm, y_treino)
print(f'Acurácia de treinamento: {knn.score(x_treino_norm, y_treino)}')

#configurar os parametros do classificador
y_pred = knn.predict(x_teste_norm)
y_pred_prob = knn.predict_proba(x_teste_norm)
acc_teste = knn.score(x_teste_norm, y_teste)
print(f"Acurácia de teste: {acc_teste}")
print("Predições para cada amostra:")
k = 1
for l, p in zip(y_pred, y_pred_prob):
    print(f"- Amostra {k}: label = {l} | probabilidades = {p}")
    k+=1

relatorio = classification_report(y_teste, y_pred, target_names=nome_das_classes)
print("Relatório de classificação:")
print(relatorio)

mat_conf = confusion_matrix(y_teste, y_pred)
print("Matriz de confusão:")
print(mat_conf)

#realizar os treinamentos
knn_pipeline = Pipeline(steps=[
    ('normalizacao', MinMaxScaler()),
    ('KNN', KNeighborsClassifier(n_neighbors=3))
])
knn_pipeline.fit(x_treino, y_treino)
y_pred = knn_pipeline.predict(x_teste)
y_pred_prob = knn_pipeline.predict_proba(x_teste)
print(f'Acurácia de treinamento: {knn_pipeline.score(x_treino, y_treino)}')

param_busca = {
    'KNN__n_neighbors': [3, 5, 7]
}
buscador = GridSearchCV(knn_pipeline, param_grid=param_busca)
buscador.fit(x, y)
print('Melhor K: ', buscador.best_params_)

pd.DataFrame.from_dict(buscador.cv_results_)

#salvar e carregar o modelo
joblib.dump(knn_pipeline, 'knn_pipeline.joblib')
knn_pipeline_carregado = joblib.load('knn_pipeline.joblib')
y_pred_prob = knn_pipeline_carregado.predict_proba(x_teste)
print(f'Acurácia de treinamentos: {knn_pipeline_carregado.score(x_treino, y_treino)}')
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


#Carregando os dados do dataset.
base = load_digits()

#fazendoa separação de treinamento e teste
previsores_train, previsores_test, classe_train, classe_test = train_test_split(base.data, base.target, test_size=0.3)

#Fazendo a classificação
classificador = LogisticRegression()
classificador.fit(previsores_train,classe_train)

predict = classificador.predict(previsores_test)

acc_dataTrain = classificador.score(previsores_train, classe_train)

acc_dataTest = classificador.score(previsores_test,classe_test) 

acc = accuracy_score(classe_test, predict)

matriz = confusion_matrix(classe_test, predict)

print(acc_dataTrain)
print(acc_dataTest)
print(acc)
print(matriz)
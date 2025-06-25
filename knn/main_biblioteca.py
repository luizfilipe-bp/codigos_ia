from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import time


def plot_metrics(y_test, y_pred, class_names, k, namefile):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Classificados')
    plt.ylabel('Verdadeiros')
    plt.title(f'{namefile}\n')
    plt.figtext(0.1, -0.15,
                f'Acurácia = {accuracy:.6f}\n'
                f'                 Precisão     Revocação\n'
                f'{class_names[0]}:     {precision[0]:.6f}  {recall[0]:.6f}\n'
                f'{class_names[1]}: {precision[1]:.6f}  {recall[1]:.6f}\n'
                f'{class_names[2]}:  {precision[2]:.6f}  {recall[2]:.6f}\n')
    plt.savefig(f'metrics_{namefile}{k}.png', bbox_inches='tight')


def run_classification(dataset, k, namefile):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, stratify=dataset.target, test_size=0.5, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    plot_metrics(y_test, y_predict, dataset.target_names, k, namefile)


def run():
    iris = load_iris()
    run_classification(iris, 1, 'iris')
    run_classification(iris, 3, 'iris')
    run_classification(iris, 5, 'iris')
    run_classification(iris, 7, 'iris')

    wine = load_wine()
    run_classification(wine, 1, 'wine')
    run_classification(wine, 3, 'wine')
    run_classification(wine, 5, 'wine')
    run_classification(wine, 7, 'wine')


if __name__ == '__main__':
    begin_time = time.time()
    run()
    end_time = time.time()
    print(f'tempo de execução sklearn: {end_time - begin_time}')

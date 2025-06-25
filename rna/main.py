import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_metrics(y_test, y_pred, class_names, namefile):
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
    plt.savefig(f'metrics_{namefile}.png', bbox_inches='tight')


def rna(dataset, namefile):
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=0)

    #pre processamento dos dados
    #dataset wine estava com resultado muito ruim sem o pre processamento
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    perceptron = MLPClassifier(max_iter=300, hidden_layer_sizes=(100, 50))
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)

    plot_metrics(y_test, y_pred, dataset.target_names, namefile)


def run():
    dataset_iris = load_iris()
    rna(dataset_iris, 'iris')
    dataset_wine = load_wine()
    rna(dataset_wine, 'wine')


if __name__ == '__main__':
    run()

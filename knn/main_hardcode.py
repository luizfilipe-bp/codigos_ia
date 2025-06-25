import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import time

def euclidian_distance(p1, p2):
    return (np.sqrt((p2['SepalLengthCm'] - p1['SepalLengthCm']) ** 2 +
                    (p2['SepalWidthCm'] - p1['SepalWidthCm']) ** 2 +
                    (p2['PetalLengthCm'] - p1['PetalLengthCm']) ** 2 +
                    (p2['PetalWidthCm'] - p1['PetalWidthCm']) ** 2))


def read_data(name_file):
    data_iris = pd.read_csv(name_file)
    training_data = pd.concat([data_iris.iloc[0:25], data_iris.iloc[50:75], data_iris.iloc[100:125]])
    training_data['DistanceTo'] = 0.0

    without_species_data = pd.concat([data_iris.iloc[25:50], data_iris.iloc[75:100], data_iris.iloc[125:150]])
    without_species_data['Species'] = ''

    data_true = pd.concat([data_iris.iloc[25:50], data_iris.iloc[75:100], data_iris.iloc[125:150]])
    return training_data, without_species_data, data_true


def point_classification(p1, training_data, k):
    for i in range(0, 75):
        d = euclidian_distance(p1, training_data.iloc[i])
        training_data.loc[training_data.index[i], 'DistanceTo'] = d

    training_data.sort_values(by='DistanceTo', inplace=True)
    neighbors = training_data.iloc[0: k]
    return neighbors['Species'].value_counts().idxmax()


def calculate_confusion_matrix(data_true, data_predicted):
    unique_species = data_true['Species'].unique()
    matrix = pd.DataFrame(index=unique_species, columns=unique_species, data=0)

    for i in range(len(data_true)):
        true_label = data_true.iloc[i]['Species']
        predicted_label = data_predicted.iloc[i]['Species']
        matrix.loc[true_label, predicted_label] += 1

    return matrix


def classify(training_data, data_without_species, k):
    for i in range(0, 75):
        classification = point_classification(data_without_species.iloc[i], training_data, k)
        data_without_species.loc[data_without_species.index[i], 'Species'] = classification


def plot_confusion_matrix(data_true, data_without_species, k):
    confusion_matrix_df = calculate_confusion_matrix(data_true, data_without_species)
    true_labels = data_true['Species']
    predicted_labels = data_without_species['Species']
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, average=None)
    recall = metrics.recall_score(true_labels, predicted_labels, average=None)    #   revocação

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Classificados')
    plt.ylabel('Verdadeiros')
    plt.title(f'Matriz de confusão KNN- K = {k} neighbors\n')
    plt.figtext(0.1, -0.15,
                f'Acurácia = {accuracy:.6f}\n'
                f'                 Precisão     Revocação\n'
                f'Setosa:     {precision[0]:.6f}  {recall[0]:.6f}\n'
                f'Versicolor: {precision[1]:.6f}  {recall[1]:.6f}\n'
                f'Virginica:  {precision[2]:.6f}  {recall[2]:.6f}\n')

    plt.savefig(f'confusion_matrix_k{k}.png', bbox_inches='tight')


def make_classification(file_name, k):
    training_data, data_without_species, data_true = read_data(file_name)
    classify(training_data, data_without_species, k)
    plot_confusion_matrix(data_true, data_without_species, k)


def run():
    make_classification('iris.csv', 1)
    make_classification('iris.csv', 3)
    make_classification('iris.csv', 5)
    make_classification('iris.csv', 7)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    begin_time = time.time()
    run()
    end_time = time.time()
    print(f'tempo de execução hardcode: {end_time - begin_time}')

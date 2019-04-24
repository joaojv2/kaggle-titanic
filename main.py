import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def process_age(dataset):
    dataset.Age = dataset.Age.fillna(-0.5)

    bins: tuple = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names: list = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(dataset.Age, bins, labels=group_names)
    dataset.Age = categories

    return dataset


def process_cabin(dataset):
    dataset.Cabin = dataset.Cabin.fillna('N')
    dataset.Cabin = dataset.Cabin.apply(lambda x: x[0])

    return dataset


def process_fares(dataset):
    dataset.Fare = dataset.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(dataset.Fare, bins, labels=group_names)
    dataset.Fare = categories

    return dataset


def process_family_size(dataset):
    dataset['FamilyS'] = dataset.SibSp + dataset.Parch

    return dataset


def process_fare_per_person(dataset):
    dataset.Fare = dataset.Fare.fillna(-0.5)
    dataset['FareP'] = dataset.Fare / (dataset.FamilyS + 1)

    return dataset


def process_age_class(dataset):
    dataset.Age = dataset.Age.fillna(-0.5)
    dataset['AgeClass'] = dataset['Age'] * dataset['Pclass']

    return dataset


def format_name(dataset):
    dataset['NameL'] = dataset.Name.apply(lambda x: x.split(' ')[0])
    dataset['NameP'] = dataset.Name.apply(lambda x: x.split(' ')[1])

    return dataset


def drop_features(dataset):
    dataset = dataset.drop(['Ticket', 'Name', 'Embarked'], axis=1)

    return dataset


def process_dataset(dataset):
    dataset = process_family_size(dataset)
    dataset = process_fare_per_person(dataset)
    dataset = process_age_class(dataset)
    dataset = process_age(dataset)
    dataset = process_cabin(dataset)
    dataset = process_fares(dataset)
    dataset = format_name(dataset)
    dataset = drop_features(dataset)
    return dataset


def encode_features(dataset_train, dataset_test):
    features: list = ['Fare', 'Cabin', 'Age', 'Sex', 'NameL', 'NameP', 'FamilyS', 'AgeClass', 'FareP']

    data_combined = pd.concat([dataset_train[features], dataset_test[features]])

    for feature in features:
        label_encoded = preprocessing.LabelEncoder()
        label_encoded = label_encoded.fit(data_combined[feature])

        dataset_train[feature] = label_encoded.transform(dataset_train[feature])
        dataset_test[feature] = label_encoded.transform(dataset_test[feature])

    return dataset_train, dataset_test


def train_test(dataset_train):
    x_all = dataset_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = dataset_train['Survived']

    num_test = 0.20
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=23)

    classifier = RandomForestClassifier()

    parameters = {
        'n_estimators': [4, 6, 9],
        'max_features': ['log2', 'sqrt', 'auto'],
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 3, 5, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }

    acc_scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(classifier, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(x_train, y_train)

    classifier = grid_obj.best_estimator_

    classifier.fit(x_train, y_train)

    return classifier


if __name__ == '__main__':
    data_train = pd.read_csv('dataset/train.csv')
    data_test = pd.read_csv('dataset/test.csv')
    data_result = pd.read_csv('dataset/gender_submission.csv')

    data_train = process_dataset(data_train)
    data_test = process_dataset(data_test)

    data_train, data_test = encode_features(data_train, data_test)

    model = train_test(data_train)

    ids = data_test['PassengerId']
    predictions = model.predict(data_test.drop('PassengerId', axis=1))

    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})

    count: int = 0
    for i in range(0, len(data_test)):
        if output['Survived'][i] == data_result['Survived'][i]:
            count += 1

    print("Hit -> {0} from {1}".format(count, len(output)))

from keras.models import Sequential
from keras.layers import Dense

from Tools.data.data_helpers import kaggle_submission

from sklearn.model_selection import StratifiedKFold, cross_val_score

def create_baseline(n_dim = 7):
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=n_dim
    , kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cross_validation_models(models, X_train, y_train, n_splits = 10):
    # Evaluate each model with 10-fold cross-validation
    results = []
    names = []
    scoring = 'f1'
    for name, model in models:
        kfold = StratifiedKFold(n_splits=n_splits, random_state=1)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def fit_predict(trained_models, X_train, y_train, X_test, to_csv = True, title = 'new_one'):
    # Fit
    for name, model in trained_models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if to_csv: kaggle_submission(predictions, title + name)

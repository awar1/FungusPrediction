import cv2 as cv
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from dictionary import create_Dictionary
from characteristics import load_characterictics

# transform a photo provided by user into cv array
def load_image_from_folder(path):
    return cv.imread(path)

def predictionTest(X,Y):
    """Returns accuracy of the used method according to KFold crossvalidation"""
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # diving data into 5 subsets and checking accuracy of each
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=k_fold)

    return scores.mean()*100

def prediction(X,Y,data):
    """predicts family of fungus based on decision tree algorithm, returns a number from 0 to 3"""
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    y_pred = clf.predict(data)
    return y_pred
def fungus_Prediction(path):
    """analyzes photo given in path and returns name of fungus"""
    image = load_image_from_folder(path)
    result = load_characterictics(image)
    X, Y = create_Dictionary()
    result2d = [result]
    wynik = prediction(X,Y,result2d)
    return{
        0:'Fusarium',
        1:'Phytophtora',
        2:'Trichoderma',
        3:'Verticillium'
    }.get(wynik[0],'none') 

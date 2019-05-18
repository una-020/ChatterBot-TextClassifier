from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pickle as pkl


def train_classifier(X, y, C):
    cls = LogisticRegression(
        random_state=0,
        C=C,
        solver='lbfgs',
        max_iter=10000,
        n_jobs=-1
    )
    cls.fit(X, y)
    return cls


def evaluate(X, yt, cls, verbose=False, name='data'):
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    if verbose:
        print("  Accuracy on %s  is: %s" % (name, acc))
    return acc


def pipeline(X, y, dX, dy, verbose=False):
    C = [0.01, 0.1, 1, 10, 100, 1000]
    max_acc = 0
    max_model = None
    max_C = None
    for c in C:
        model = train_classifier(X, y, c)
        acc = evaluate(dX, dy, model, verbose=verbose, name=("Dev C = %s" % c))
        if acc > max_acc:
            max_acc = acc
            max_model = model
            max_C = c
    if verbose:
        print("Best C = %s" % max_C)
        evaluate(X, y, max_model, verbose=True, name="Train")
        evaluate(dX, dy, max_model, verbose=True, name="Dev")
    return max_model, max_acc, max_C


def main():
    X = pkl.load(open("X.pkl", "rb"))
    Y = pkl.load(open("Y.pkl", "rb"))
    kf = KFold(n_splits=4, shuffle=True)
    avg_acc = 0
    avg_cnt = 0
    for train_index, test_index in kf.split(X):
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = Y[train_index], Y[test_index]
        cls = train_classifier(trainX, trainY, 1)
        print("Evaluating Split", avg_cnt + 1)
        acc = evaluate(testX, testY, cls, verbose=True, name="Test")
        avg_acc += acc
        avg_cnt += 1
    print("Average Accuracy", avg_acc / avg_cnt)


if __name__ == "__main__":
    main()

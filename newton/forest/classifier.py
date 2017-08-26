import numpy as np
from sklearn.externals import joblib

class Forest:

    def __init__(self, area_id):
        self.area_id = area_id
        self.classifier = joblib.load('serialized/{}_data'.format(area_id))

    def query(self, points, n_results):
        predicted = self.classifier.predict_proba(points)
        result = []
        for i in range(predicted.shape[0]):
            best = np.argpartition(predicted[i],-n_results)[-n_results:]
            result.append(best)
        return result

    def get_class(self,index):
        return self.classifier.classes_[index]


if __name__ == '__main__':

    rf = Forest(10)
    q = rf.query([[800,700,0,700,720]],4)
    print(q)
    print(rf.get_class(q))
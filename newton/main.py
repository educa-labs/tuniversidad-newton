from forest import *
from knn import *
import numpy as np


class Newton:

    def __init__(self, area_ids,serialized_forests,serialized_tree, data_dir, n_forest_results=3, k=5):
        self.area_ids = area_ids
        self.balltree = Tree(serialized_tree,data_dir)
        self.active_forests = {x: None for x in area_ids}
        self.n_forest_results = n_forest_results
        self.k = k
        self.serialized_forests = serialized_forests

    def get_recs(self, area_id, scores):
        if self.active_forests[area_id] is None:
            self.active_forests[area_id] = Forest(area_id, self.serialized_forests)
        forest = self.active_forests[area_id]
        prediction = forest.get_class(forest.query(scores,self.n_forest_results))
        recommendations = []
        for carreer_set in prediction:
            recommendations.append(self.balltree.query(carreer_set,self.k))
        return np.array(recommendations)



    def filter_recs(self, user, carreers):
        pass


if __name__ == '__main__':
    sisrec = Newton(np.array([i for i in range(1, 12)]),'forest/serialized','knn/serialized','data',3,5)
    print(sisrec.get_recs(1, [[800,700,0,700,720], [800,700,0,700,720]]))
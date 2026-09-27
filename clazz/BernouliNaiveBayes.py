import numpy as np

class BernouliNaiveBayes:

    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.classes = None
        self.class_prior = {}
        self.feature_prob = {}
        
    def fit(self, X, y):

        self.X = np.array(X)
        self.y = np.array(y)

        n_samples, n_features = X.shape

        self.classes = np.unique(y)

        for clazz in self.classes:

            clazz = int(clazz)
            clazz_samples = np.sum(self.y == clazz)

            # P(class)
            self.class_prior[clazz] = clazz_samples / n_samples

            self.feature_prob[clazz] = np.zeros(n_features)

            # P(feature = 1 | class)        
            for feature in range(n_features):

                feature_count = np.sum(
                    X[self.y == clazz, feature]
                )
                self.feature_prob[clazz][feature] = (feature_count + self.alpha) / (clazz_samples + (2 * self.alpha))



    def predict(self, X):

        predictions = []

        n_samples, n_features = X.shape

        for i in range(n_samples):

            class_scores = {}

            for clazz in self.classes:

                clazz = int(clazz)

                propability_score = np.log(self.class_prior[clazz])

                for feature in range(n_features):

                    x_i = X[i, feature]

                    feature_prob_i_in_class = self.feature_prob[clazz][feature]
                    propability_score += (x_i * np.log(feature_prob_i_in_class) + (np.log(1 - feature_prob_i_in_class)*(1 - x_i)))

                class_scores[clazz] = propability_score

            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)
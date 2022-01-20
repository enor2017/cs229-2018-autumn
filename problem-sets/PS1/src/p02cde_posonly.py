import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col = 't', add_intercept = True)
    model = LogisticReg()
    model.fit(x_train, t_train)
    util.plot(x_train, t_train, theta = model.theta, save_path = "output/p02c_train")
    print(model.theta)

    # apply on test set
    x_test, t_test = util.load_dataset(test_path, label_col = 't', add_intercept = True)
    t_pred = model.predict(x_test)
    util.plot(x_test, t_test, theta = model.theta, save_path = "output/p02c_test")
    np.savetxt(pred_path_c, t_pred, fmt = '%d')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    model_d = LogisticReg()
    model_d.fit(x_train, y_train)
    util.plot(x_train, y_train, theta = model.theta, save_path = "output/p02d_train")
    print(model_d.theta)

    # apply on test set
    x_test, y_test = util.load_dataset(test_path, add_intercept = True)
    util.plot(x_test, y_test, theta = model_d.theta, save_path = "output/p02d_test")
    y_pred = model_d.predict(x_test)
    np.savetxt(pred_path_d, y_pred, fmt='%d')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


class LogisticReg:
    def __init__(self):
        self.theta = None

    def get_hypothesis(self, theta, x):
        """

        Args:
            theta: shape (n, )
            x: shape (m, n)

        Returns:
            hypothesis for all data, (m, )
        """
        return 1 / (1 + np.exp(-x @ theta))

    def get_gradient(self, theta, x, y):
        """

        Args:
            theta: shape (n, )
            x: shape (m, n)
            y: shape (m, )

        Returns:
            gradient, shape (n, )
        """
        m, n = x.shape
        return - 1 / m * (x.T @ (y - self.get_hypothesis(theta, x)))

    def get_hessian(self, theta, x):
        m, n = x.shape
        hyp = self.get_hypothesis(theta, x)
        # reshape hyp to (m, 1)
        hyp = np.reshape(hyp, (m, 1))
        # here broadcast rule applies          *
        return 1 / m * (x.T @ (hyp * (1 - hyp) * x))

    def fit(self, x, y):
        epsilon = 10 ** (-5)
        m, n = x.shape

        # initialize to all zero
        self.theta = np.zeros(n)
        while True:
            new_theta = self.theta.copy()
            new_theta -= np.linalg.inv(self.get_hessian(new_theta, x)) @ self.get_gradient(new_theta, x, y)
            delta = np.linalg.norm(new_theta - self.theta)
            # always update theta
            self.theta = new_theta
            if delta < epsilon:
                break

    def predict(self, x):
        """

        Args:
            x: shape (m, n)

        Returns:
            all binary predictions, shape (m, )
        """
        return x @ self.theta >= 0

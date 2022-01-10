import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # train logistic regression
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # plot data and decision boundary
    util.plot(x_train, y_train, save_path = "output/p01b_{}.png".format(pred_path[-5]), theta = clf.theta)

    # print predictions
    print("Theta is: ", clf.theta)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == y_train))
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt = '%d')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self):
        super().__init__()
        self.theta = None

    def get_sigmoid(self, theta, x):
        """
        Calculate the sigmoid function h_{theta}(x) = g(theta^T * x) = 1 / (1 + exp{-theta^T * x})
        Args:
            theta: parameter theta, with shape (n, )
            x: all data set, with shape (m, n)

        Returns: The hypothesis vector for all data, with shape (m, )

        """
        # here matrix multiplication (x * theta) returns an m * 1 matrix, which is the hypothesis
        # of all data
        # print(f"x shape: {x.shape}, theta shape: {theta.shape}, @: {(x@theta).shape}, dot: {np.dot(x, theta).shape}")
        return 1 / (1 + np.exp(- x @ theta))

    def get_hessian(self, theta, x):
        """

        Args:
            theta: parameter theta, with shape (n, )
            x: all data set, with shape (m, n)

        Returns: the hessian matrix, shape (n, n)

        """
        m, _ = x.shape
        g = self.get_sigmoid(theta, x)
        g = np.reshape(g, (m, 1))   # g was a (m, ) vector, re-shape to (m, 1) matrix
        return 1 / m * (x.T @ (g * (1 - g) * x))

    def get_gradient(self, theta, x, y):
        """

        Args:
            theta: parameter theta,  with shape (n, )
            x: all data set, with shape (m, n)
            y: all data set, with shape (m, )

        Returns: the gradient of J(theta), shape (n, )

        """
        m, _ = x.shape
        return -1 / m * (x.T @ (y - self.get_sigmoid(theta, x)))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        self.theta = np.zeros(n)
        while True:
            new_theta = self.theta.copy()
            # update rule: theta := theta - H^{-1} * grad J(theta)
            new_theta -= np.linalg.inv(self.get_hessian(new_theta, x)) @ \
                         self.get_gradient(new_theta, x, y)

            # terminate if error less than epsilon 10^{-5}
            diff = np.linalg.norm(new_theta - self.theta)
            self.theta = new_theta  # update new theta
            if diff < 10 ** (-5):
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        # if z >= 0, g(z) >= 0.5, we predict 1
        return x @ self.theta >= 0
        # alternatively, we can directly check if g(z) >= 0.5
        # return self.get_sigmoid(self.theta, x) >= 0.5

        # *** END CODE HERE ***

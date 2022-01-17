import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    # plot data and decision boundary
    # util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    # save predictions
    # x_eval, y_eval = util.load_dataset(eval_path)
    # y_pred = model.predict(x_eval)
    # np.savetxt(pred_path, y_pred > 0.5, fmt = '%d')

    # question (f)
    util.plot(x_train, y_train, model.theta, save_path= 'output/p01f_GDA_{}.png'.format(pred_path[-5]))

    # question (h): only apply to dataset 1
    if pred_path[-5] == '1':
        x_train[:, 1] = np.log(x_train[:, 1])
        model.fit(x_train, y_train)
        util.plot(x_train, y_train, model.theta, save_path= 'output/p01h_GDA_{}.png'.format(pred_path[-5]))

    # *** END CODE HERE ***

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self):
        super().__init__()
        # theta has shape (n, )
        self.theta = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape

        phi = 1 / m * np.sum(y)
        mu0 = (x.T @ (1 - y)) / np.sum(1 - y)
        mu1 = (x.T @ y) / np.sum(y)

        # mu_y has shape (m, n), each row is mu_{y^{i}},
        # so we need to extend mu for m times, choose proper mu{0, 1} for each row
        # we can use '*' to do this:
        matrix_y = np.reshape(y, (m, -1))
        # use '*' to multiply a matrix/vector A with shape (a, 1) and a vector B with shape (b, )
        # results in a matrix with shape (a, b), which repeat A for b times, i.e, column i is
        # B_{i} * A,
        # example: if A = [[2], [3], [4]], B = [[5], [7]],
        # then A * B = [[10, 14], [15, 21], [20, 28]]
        # this is equivalent to A @ B.T, where B is reshaped to (b, 1)
        mu_y = matrix_y * mu1 + (1 - matrix_y) * mu0

        x_centered = x - mu_y

        # notice in ps, the x matrix is (n, m), but in coding, the x is (m, n)
        # so the transpose is reversed.
        sigma = 1 / m * x_centered.T @ x_centered

        # calculate theta
        inv_sigma = np.linalg.inv(sigma)
        theta = inv_sigma @ (mu1 - mu0)
        theta_0 = 1 / 2 * (mu0.T @ inv_sigma @ mu0 - mu1.T @ inv_sigma @ mu1) - np.log((1 - phi) / phi)
        # concatenate two rows
        theta_0 = np.reshape(theta_0, (1, ))
        self.theta = np.concatenate((theta_0, theta), axis = 0)
        # print(f"theta: {theta.shape}, 0: {theta_0.shape}, self: {self.theta.shape}")

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return util.add_intercept(x) @ self.theta >= 0
        # *** END CODE HERE

import pandas as pd
import numpy as np
import xgboost
from scipy.special import expit, logit
from sklearn import metrics

np.random.seed(0)

class xgbConstrTrainer:
    def __init__(
        self,
        n_instances=1000,
        n_vars=5,
        X_noise=0,
        Y_noise=1,
    ) -> None:

        self.n_instances = n_instances
        self.n_vars = n_vars
        self.X_noise = X_noise
        self.Y_noise = Y_noise
        self.xgboost_params = {}
        self.metrics_dict = {
            "MSE": metrics.mean_squared_error,
            "MAE": metrics.mean_absolute_error,
            "MAPE": metrics.mean_absolute_percentage_error,
        }

    def load_causalml_mode_1(self, adj=0.0):
        # Skeleton from CausalML: def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
        """Synthetic data with a difficult nuisance components and an easy treatment effect
            From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
        Args:
            n (int, optional): number of observations
            p (int optional): number of covariates (>=5)
            Y_noise (float): standard deviation of the error term
            adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
        Returns:
            (tuple): Synthetically generated samples with the following outputs:
                - y ((n,)-array): outcome variable.
                - X ((n,p)-ndarray): independent variables.
                - w ((n,)-array): treatment flag with value 0 or 1.
                - tau ((n,)-array): individual treatment effect.
                - b ((n,)-array): expected outcome.
                - e ((n,)-array): propensity of receiving treatment.
        """
        n = self.n_instances * 2
        p = self.n_vars
        X_noise = self.X_noise
        Y_noise = self.Y_noise

        X = (
            np.random.uniform(size=n * p) + np.random.normal(0, X_noise, size=n * p)
        ).reshape((n, -1))

        b = (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
        )
        eta = 0.1
        e = np.maximum(
            np.repeat(eta, n),
            np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
        )
        e = expit(logit(e) - adj)
        tau = (X[:, 0] + X[:, 1]) / 2

        w = np.random.binomial(1, e, size=n)
        y = b + (w - 0.5) * tau + Y_noise * np.random.normal(size=n)

        train_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        train_df["T"] = w
        train_df["Y"] = y

        self.X_features = [f"col_{i}" for i in range(X.shape[1])] + ["T"]
        self.Y_feature = ["Y"]
        self.df = train_df.iloc[: self.n_instances, :].reset_index()
        self.test_df = train_df.iloc[self.n_instances :, :].reset_index()
        self.train_ite = tau[: self.n_instances]
        self.test_ite = tau[self.n_instances :]

    @staticmethod
    def load_predefined_causalml_mode_1_dag():
        dag = [
            ["col_0", "col_1", "T"],
            ["col_2"],
            ["col_3"],
            ["col_4"],
            ["col_0"],
            ["col_1"],
        ]
        return dag

    def add_edges_constraints(self, dag_edges):

        self.dag_edges = dag_edges
        # Duplicate columns to fit to dag constraints
        used_vars = []
        copies_used = 1
        for i in range(len(self.dag_edges)):
            for j in range(len(self.dag_edges[i])):

                if self.dag_edges[i][j] in used_vars:
                    copy_name = self.dag_edges[i][j] + "constr" + str(copies_used)
                    self.df[copy_name] = self.df[self.dag_edges[i][j]].copy()
                    self.test_df[copy_name] = self.test_df[self.dag_edges[i][j]].copy()
                    self.X_features.append(copy_name)
                    self.dag_edges[i][j] = copy_name
                    copies_used += 1

                else:
                    used_vars.append(self.dag_edges[i][j])

        # Set xgb param
        xgb_interaction_constraints = self.dag_edges
        self.xgboost_params["interaction_constraints"] = xgb_interaction_constraints

    @staticmethod
    def train_xgb_native(X, y, params):
        dmatrix = xgboost.DMatrix(X, label=y)
        model = xgboost.train(params, dmatrix)
        return model

    @staticmethod
    def predict_xgb_native(model, X):
        dmatrix = xgboost.DMatrix(X)
        return model.predict(dmatrix)

    def slearn_xgboost(self, T="T"):

        params = self.xgboost_params.copy()

        # Train and predict model 0
        df = self.df.copy()
        test_df = self.test_df.copy()

        m = self.train_xgb_native(df[self.X_features], df[self.Y_feature], params)

        T_cols = [c for c in test_df.columns if "Tconstr" in c] + [T]
        for c in T_cols:
            test_df[c] = 0

        preds0 = self.predict_xgb_native(m, test_df[self.X_features])

        for c in T_cols:
            test_df[c] = 1

        preds1 = self.predict_xgb_native(m, test_df[self.X_features])

        return preds1 - preds0, m

    def run(self):

        # Load data ------------------------------------------------------------
        self.load_causalml_mode_1()

        # Eval unconstrained model ---------------------------------------------
        print("XGB Unconstrained")
        real_ITE = self.test_ite
        # SLEARN
        pred_ITE, model_unconstr = self.slearn_xgboost(T="T")
        metric_values = dict((k, []) for k in self.metrics_dict.keys())
        for k, v in self.metrics_dict.items():
            metric_values[k] = np.round(v(real_ITE, pred_ITE), 4)
        print(metric_values)

        # Eval constrained model -----------------------------------------------
        print("XGB Constrained")
        dag_edges = self.load_predefined_causalml_mode_1_dag()
        self.add_edges_constraints(dag_edges)
        # SLEARN
        pred_ITE, constr = self.slearn_xgboost(T="T")
        metric_values = dict((k, []) for k in self.metrics_dict.keys())
        for k, v in self.metrics_dict.items():
            metric_values[k] = np.round(v(real_ITE, pred_ITE), 4)
        print(metric_values)


if __name__ == "__main__":

    model_trainer = xgbConstrTrainer(
        n_instances=1000,
        n_vars=5,
        X_noise=0,
        Y_noise=1,
    )

    model_trainer.run()


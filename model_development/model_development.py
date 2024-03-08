# %%
from sklearn.model_selection import train_test_split
import pandas as pd

high_correlations = pd.read_csv('high_correlations.csv')
target_variable = high_correlations['market_value_capped']
high_correlations = high_correlations.drop(columns=['market_value_capped'])
X_train, X_test, y_train, y_test = train_test_split(high_correlations, target_variable, test_size=0.2, random_state=42)


pca_10principal = pd.read_csv('pca_10component.csv')
target_variable = pca_10principal['market_value_capped']
pca_10principal = pca_10principal.drop(columns=['market_value_capped'])
PCA_X_train, PCA_X_test, PCA_y_train, PCA_y_test = train_test_split(pca_10principal, target_variable, test_size=0.2, random_state=42)

# %% [markdown]
# # Naively Applying Regression Models

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
print("HIGH_CORR_LIN_MODEL:", lin_model.score(X_test, y_test))
print("HIGH_CORR_LIN_MODEL RMSE:", root_mean_squared_error(lin_model.predict(X_test), y_test))

pca_lin_model = LinearRegression()
pca_lin_model.fit(PCA_X_train, PCA_y_train)
print("PCA_LIN_MODEL:", pca_lin_model.score(PCA_X_test, PCA_y_test))
print("PCA_LIN_MODEL RMSE:", root_mean_squared_error(pca_lin_model.predict(PCA_X_test), PCA_y_test))

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

decision_model = DecisionTreeRegressor()
decision_model.fit(X_train, y_train)
print("HIGH_CORR_DEC_MODEL:", decision_model.score(X_test, y_test))
print("HIGH_CORR_DEC_MODEL RMSE:", root_mean_squared_error(decision_model.predict(X_test), y_test))

pca_decision_model = DecisionTreeRegressor()
pca_decision_model.fit(PCA_X_train, PCA_y_train)
print("PCA_DEC_MODEL:", pca_decision_model.score(PCA_X_test, PCA_y_test))
print("PCA_DEC_MODEL RMSE:", root_mean_squared_error(pca_decision_model.predict(PCA_X_test), PCA_y_test))

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
print("HIGH_CORR_FOREST_MODEL:", forest_model.score(X_test, y_test))
print("HIGH_CORR_FOREST_MODEL RMSE:", root_mean_squared_error(forest_model.predict(X_test), y_test))

pca_forest_model = RandomForestRegressor()
pca_forest_model.fit(PCA_X_train, PCA_y_train)
print("PCA_FOREST_MODEL:", pca_forest_model.score(PCA_X_test, PCA_y_test))
print("PCA_FOREST_MODEL RMSE:", root_mean_squared_error(pca_forest_model.predict(PCA_X_test), PCA_y_test))

# %%
from sklearn.ensemble import GradientBoostingRegressor

grad_boost_model = GradientBoostingRegressor()
grad_boost_model.fit(X_train, y_train)
print("HIGH_CORR_GRAD_BOOST_MODEL:", grad_boost_model.score(X_test, y_test))
print("HIGH_CORR_GRAD_BOOST_MODEL RMSE:", root_mean_squared_error(grad_boost_model.predict(X_test), y_test))

pca_grad_boost_model = GradientBoostingRegressor()
pca_grad_boost_model.fit(PCA_X_train, PCA_y_train)
print("PCA_GRAD_BOOST_MODEL:", pca_grad_boost_model.score(PCA_X_test, PCA_y_test))
print("PCA_GRAD_BOOST_MODEL RMSE:", root_mean_squared_error(pca_grad_boost_model.predict(PCA_X_test), PCA_y_test))

# %%
import xgboost

xgb_model = xgboost.XGBRegressor()
xgb_model.fit(X_train, y_train)
print("HIGH_CORR_XGB_MODEL:", xgb_model.score(X_test, y_test))
print("HIGH_CORR_XGB_MODEL RMSE:", root_mean_squared_error(xgb_model.predict(X_test), y_test))

pca_xgb_model = xgboost.XGBRegressor()
pca_xgb_model.fit(PCA_X_train, PCA_y_train)
print("PCA_XGB_MODEL:", pca_xgb_model.score(PCA_X_test, PCA_y_test))
print("PCA_XGB_MODEL RMSE:", root_mean_squared_error(pca_xgb_model.predict(PCA_X_test), PCA_y_test))



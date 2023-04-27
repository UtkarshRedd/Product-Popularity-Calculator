import os
import sys
import pandas as pd
import numpy as np
import config as cfg
#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold, StratifiedKFold
from xgboost import XGBRegressor
from xgboost import plot_importance


from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy
from pathlib import Path
from cassandra.auth import PlainTextAuthProvider

# Preprocessing function
def preprocessing(clicks_path, wishlist_path, cart_path, orders_completed_path):
    
    #### Clicks ####
    df_clicks =  pd.read_csv(clicks_path)
    df_clicks.rename(columns={'f0_': 'product_id'}, inplace=True)
    df_clicks = df_clicks[['id', 'brand', 'category', 'event', 'event_text', 'name', 'original_timestamp', 'timestamp',\
                       'price', 'product_id', 'price']]
    
    # Brand wise aggregation
    df_clicks_product = df_clicks.copy()
    df_clicks_product['clicks'] = np.zeros(len(df_clicks_product))
    df_clicks_product = df_clicks_product.groupby('product_id', as_index=False).agg({'clicks': 'count'})
    df_clicks_product['product_id'] = df_clicks_product['product_id'].astype(int)


    # Mapping brands to categories
    brand_to_category_dict = dict(zip(df_clicks['brand'], df_clicks['category']))

    #### Orders completed ####
    df_order_completed = pd.read_csv(orders_completed_path)
    df_order_completed = df_order_completed[df_order_completed['products'].isnull() == False]
    df_order_completed = pd.concat([pd.DataFrame(eval(x)) for x in df_order_completed['products']], \
                               keys=df_order_completed['original_timestamp']).reset_index(level=1, drop=True).reset_index()
    df_order_completed = df_order_completed.fillna('0')
    for i, row in df_order_completed.iterrows():
        if (row['brand'] == '0'):
            row['brand'] = row['item_brand']
        if (row['name'] == '0'):
            row['name'] = row['item_name']
        if (row['product_id'] == '0'):
            row['product_id'] = row['item_id']
        if (row['category'] == '0'):
            if (row['brand'] in brand_to_category_dict.keys()):
                row['category'] = brand_to_category_dict[row['brand']]
        if (row['price'] == '0'):
            row['price'] = row['item_price']
        if (row['quantity'] == '0'):
            row['quantity'] = row['item_quantity']
    
    df_order_completed.drop(['item_brand', 'item_id', 'item_name', 'coupon', 'currency', 'category_id', 'image_url', 'size', 'item_price', 'item_quantity'], axis=1, inplace=True)
    df_order_completed['price'] = df_order_completed['price'].astype(float)
    df_order_completed['quantity'] = df_order_completed['quantity'].astype(float)
    df_order_completed['product_id'] = df_order_completed['product_id'].astype(int)

    df_order_completed['quantity'] = df_order_completed['quantity'].apply(lambda x: 1 if x == 0 else x)
    df_order_completed['category'] = df_order_completed['category'].apply(lambda x: 'not set' if x == '0' else x)

    # Brand wise aggregation
    df_order_completed_product = df_order_completed.copy()
    df_order_completed_product['no_of_unique_purchases'] = np.zeros(len(df_order_completed_product))

    df_grouped = df_order_completed_product.groupby('product_id', as_index=False).agg({'price':'sum', 'quantity':'sum', 'no_of_unique_purchases': 'count'})
    df_grouped.rename(columns={'quantity':'no_of_purchases'}, inplace=True)
    df_grouped.sort_values(by='no_of_unique_purchases', ascending=False, inplace=True)
    df_order_completed_product = df_grouped.copy()

    ### Wishlist ###
    df_wishlist_cart = pd.read_csv(wishlist_path)
    df_wishlist_cart.rename(columns={'f0_': 'product_id'}, inplace=True)

    # Brand wise aggregation
    df_wishlist = df_wishlist_cart
    df_wishlist_product = df_wishlist.copy()
    df_wishlist_product['no_of_adds_to_wishlist'] = np.zeros(len(df_wishlist_product))
    df_grouped = df_wishlist_product.groupby('product_id', as_index=False)[['brand', 'no_of_adds_to_wishlist', 'price']].\
        agg({'price': 'sum', 'no_of_adds_to_wishlist': 'count'}).sort_values('no_of_adds_to_wishlist', ascending= False)
    df_grouped['product_id'] = df_grouped['product_id'].astype(int)
    df_wishlist_product = df_grouped.copy()

    ### Cart ###
    df_cart = pd.read_csv(cart_path)
    df_cart_product = df_cart.copy()
    df_cart_product['no_of_adds_to_cart'] = np.zeros(len(df_cart_product))
    df_cart_product = df_cart_product.groupby('product_id', as_index=False)[['category', 'price', 'quantity', 'no_of_adds_to_cart']]\
        .agg({'price': 'sum', 'quantity': 'sum', 'no_of_adds_to_cart': 'count'}).sort_values('no_of_adds_to_cart', ascending=False)
    df_cart_product['product_id'] = df_cart_product['product_id'].astype(int)
    

    ### Joining All tables ###
    # Brand Tables
    df_joined_product = df_clicks_product.merge(df_order_completed_product, on='product_id', how='inner', suffixes=('', '_order_completes')).\
                            merge(df_wishlist_product, on='product_id', how='inner', suffixes=('', '_wishlist'))
    df_joined_product = df_joined_product.merge(df_cart_product, on='product_id', how='inner', suffixes=('', '_cart'))
    df_joined_product.drop(['price', 'price_wishlist', 'price_cart', 'quantity'], axis=1, inplace=True)
    df_ml = df_joined_product[['product_id', 'clicks', 'no_of_adds_to_wishlist', 'no_of_adds_to_cart', 'no_of_unique_purchases']].copy()
    
    normalize_cols = ['clicks', 'no_of_adds_to_wishlist', 'no_of_adds_to_cart', 'no_of_unique_purchases']

    for column in normalize_cols:
        df_ml[column] = (df_ml[column] - df_ml[column].min()) / (df_ml[column].max() - df_ml[column].min())    
    df_ml = df_ml.dropna()
    df_ml.rename(columns={'no_of_adds_to_wishlist': 'adds_to_cart_from_wishlist'}, inplace=True)
    df_joined_product.rename(columns={'no_of_adds_to_wishlist': 'adds_to_cart_from_wishlist'}, inplace=True)
    return df_ml, df_joined_product

def train_compare_models():

    df_ml, df_joined_product = preprocessing(cfg.clicks_path, cfg.wishlist_path, cfg.cart_path, cfg.order_completed_path)
    #df_ml = df_ml.dropna()
    X = df_ml[['clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart']]
    Y = df_ml['no_of_unique_purchases']
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                        test_size = 0.3, random_state = 123)
    

    #### Random Forest ####
    rf_reg = RandomForestRegressor()
    rf_reg.fit(x_train, y_train)
    train_score = rf_reg.score(x_train, y_train)
    print(train_score)
    rf_regressor = RandomForestRegressor(random_state = 123)

    params = {
        'bootstrap': [True],
        'max_depth': [None, 80],
        'max_features': [1, 2],
        'min_samples_leaf': [1, 3],
        'min_samples_split': [2, 8],
        'n_estimators': [100, 1000]
    }

    print('Fitting and tuning Random Forest')
    tuned_rf_regressor = GridSearchCV(estimator=rf_regressor, 
                   param_grid=params,
                   scoring='r2', 
                   verbose=1, cv=15)


    tuned_rf_regressor.fit(x_train, y_train)
    print("Best parameters:", tuned_rf_regressor.best_params_)
    #print("\nLowest MAE: ", (-tuned_xgb.best_score_)**(1/2.0))
    print("\nR2: ", tuned_rf_regressor.best_score_)
    rf_r2 = tuned_rf_regressor.best_score_

    #  Predicting the Test set results
    y_pred = tuned_rf_regressor.best_estimator_.predict(x_test)
    rf_test_R2 = r2_score(y_test, y_pred)
    print("\nR2 score for RF on test data: ", rf_test_R2)

    # Plotting the feature importance for RF
    importances = tuned_rf_regressor.best_estimator_.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    plt.figsize=(6,6)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f'{cfg.PLOTS_DIR}/RF_Feature_Importance.png', bbox_inches='tight')

    plt.show()


    importances_rf = importances.copy()
    rf_imp = pd.DataFrame({'features': tuned_rf_regressor.best_estimator_.feature_names_in_, 'importance': tuned_rf_regressor.best_estimator_.feature_importances_})
    rf_imp.set_index('features', inplace=True)
    print(rf_imp)


    ### XGBoost ####
    print("Fitting and tuning XGBoost")
    xgb = XGBRegressor(seed = 123)

    params = { 
        'max_depth': [3,6,10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.3, 0.7]
    }


    tuned_xgb = GridSearchCV(estimator=xgb, 
                    param_grid=params,
                    scoring='r2', 
                    verbose=1, cv=10)


    tuned_xgb.fit(x_train, y_train)
    print("Best parameters:", tuned_xgb.best_params_)
    #print("\nLowest MAE: ", (-tuned_xgb.best_score_)**(1/2.0))
    print("\R2 for XGB best estimator: ", tuned_xgb.best_score_)
    xgb_mae = tuned_xgb.best_score_

    # Predicting the Test set results
    y_preds = tuned_xgb.best_estimator_.predict(x_test)
    xgb_test_accuracy = mae(y_test, y_preds)
    xgb_test_R2 = r2_score(y_test, y_preds)
    print(f"MAE on the testing data: {xgb_test_accuracy}\n R2 score on testing data: {xgb_test_R2}")

    # Plotting xgb feature importance
    importances = tuned_xgb.best_estimator_.feature_importances_
    indices = np.argsort(importances)
    features = tuned_xgb.best_estimator_.feature_names_in_

    plt.figsize = (6,6)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f"{cfg.PLOTS_DIR}/XGB_feature_importance.jpg", bbox_inches='tight')

    plt.show()

    importances_xgb = importances.copy()
    xgb_imp = pd.DataFrame({'features': tuned_xgb.best_estimator_.feature_names_in_, 'importance': tuned_xgb.best_estimator_.feature_importances_})
    xgb_imp
    xgb_imp.set_index('features', inplace=True)

    print(xgb_imp)
    # xgb_test_accuracy = 100

    ######## Fitting Huber Regressor ########
    hbr = HuberRegressor()
    print("Fitting and tuning Huber Regressor")
    params = { 
        'alpha': [0.0001, 0.001, 0.01],
        'epsilon': [1.35, 1.20, 1.4],
        'max_iter': [100, 500],
        'tol': [1e-05, 1e-04, 1e-03]
    }


    tuned_hbr = GridSearchCV(estimator=hbr, 
                    param_grid=params,
                    scoring='r2', 
                    verbose=1, cv=10)


    tuned_hbr.fit(x_train, y_train)
    print("Best parameters:", tuned_hbr.best_params_)
    #print("\nLowest MAE: ", (-tuned_xgb.best_score_)**(1/2.0))
    print("R2: ", tuned_hbr.best_score_)

    y_preds = tuned_hbr.best_estimator_.predict(x_test)
    hbr_test_accuracy = mae(y_test, y_preds)
    hbr_test_R2 = r2_score(y_test, y_preds)
    print(f"MAE on the testing data: {hbr_test_accuracy}\n R2 score: {hbr_test_R2}")

    importances = tuned_hbr.best_estimator_.coef_
    indices = np.argsort(importances)
    features = X.columns
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f"{cfg.PLOTS_DIR}/Huber_feature_importance.jpg", bbox_inches='tight')
    plt.show()

    importance_df_huber = pd.DataFrame({'feature': X.columns, 'weights': tuned_hbr.best_estimator_.coef_})
    importance_df_huber.sort_values('weights', ascending=False, inplace=True)
    importance_df_huber['weights'] = importance_df_huber['weights']/importance_df_huber['weights'].sum()
    importance_df_huber.set_index('feature', inplace=True)
    print(importance_df_huber)
    dict1 = {}
    dict1['RandomForest'] = abs(rf_test_R2)
    dict1['XGBoost'] = abs(xgb_test_R2)
    dict1['HuberRegressor'] = hbr_test_R2
    df_results = pd.DataFrame.from_dict(dict1, orient='index', columns=['R2'])

    print("Model with best R2: ", df_results['R2'].idxmax())

    result = df_results['R2'].idxmax()
    if (result == 'RandomForest'):
        print("Random Forest is giving better performance on Regression")
        df_X = df_joined_product[['product_id', 'clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']].copy()
        normalize_cols = ['clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']

        for column in normalize_cols:
            df_X[column] = (df_X[column] - df_X[column].min()) / (df_X[column].max() - df_X[column].min())    

        if (max(rf_imp['importance']) > 0.5):
            w_purchase = max(rf_imp['importance']) - 0.15
        else:
            w_purchase = max(rf_imp['importance']) + 0.15
        w_clicks = rf_imp.loc['clicks', 'importance'] * (1 - w_purchase)
        w_wishlist = rf_imp.loc['adds_to_cart_from_wishlist', 'importance'] * (1 - w_purchase)
        w_cart = rf_imp.loc['no_of_adds_to_cart', 'importance'] * (1 - w_purchase)
        df_popularity_fin = df_X
        df_popularity_fin['popularity_score'] = (df_X['no_of_purchases'] * w_purchase + df_X['clicks'] * w_clicks + df_X['adds_to_cart_from_wishlist'] * w_wishlist + df_X['no_of_adds_to_cart'] * w_cart)/(w_purchase + w_clicks + w_wishlist + w_cart)
        df_popularity_fin.sort_values('popularity_score', ascending=False, inplace=True)
        df_popularity_fin.to_csv('Output/popularity_score.csv', index=False)

        return df_popularity_fin

    elif(result == 'XGBoost'):
        print("XGBoost is giving better performance on Regression")
        df_X = df_joined_product[['product_id', 'clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']].copy()

        normalize_cols = ['clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']

        for column in normalize_cols:
            df_X[column] = (df_X[column] - df_X[column].min()) / (df_X[column].max() - df_X[column].min())    

        if (max(xgb_imp['importance']) > 0.5):
            w_purchase = max(xgb_imp['importance']) - 0.15
        else:
            w_purchase = max(xgb_imp['importance']) + 0.15
        w_clicks = xgb_imp.loc['clicks', 'importance'] * (1 - w_purchase)
        w_wishlist = xgb_imp.loc['adds_to_cart_from_wishlist', 'importance'] * (1 - w_purchase)
        w_cart = xgb_imp.loc['no_of_adds_to_cart', 'importance'] * (1 - w_purchase)
        df_popularity_fin = df_X

        df_popularity_fin['popularity_score'] = (df_X['no_of_purchases'] * w_purchase + df_X['clicks'] * w_clicks + df_X['adds_to_cart_from_wishlist'] * w_wishlist + df_X['no_of_adds_to_cart'] * w_cart)/(w_purchase + w_clicks + w_wishlist + w_cart)
        df_popularity_fin.sort_values('popularity_score', ascending=False, inplace=True)
        df_popularity_fin.to_csv(f'{cfg.OUTPUT_DIR}/popularity_score.csv', index=False)

        return df_popularity_fin
    
    else:
        print("HuberRegressor is giving better performance on Regression")
        df_X = df_joined_product[['product_id', 'clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']].copy()

        normalize_cols = ['clicks', 'adds_to_cart_from_wishlist', 'no_of_adds_to_cart', 'no_of_purchases']

        for column in normalize_cols:
            df_X[column] = (df_X[column] - df_X[column].min()) / (df_X[column].max() - df_X[column].min())    

        if (max(importance_df_huber['weights']) > 0.5):
            w_purchase = max(importance_df_huber['weights']) - 0.15
        else:
            w_purchase = max(importance_df_huber['weights']) + 0.15
        w_clicks = importance_df_huber.loc['clicks', 'weights'] * (1 - w_purchase)
        w_wishlist = importance_df_huber.loc['adds_to_cart_from_wishlist', 'weights'] * (1 - w_purchase)
        w_cart = importance_df_huber.loc['no_of_adds_to_cart', 'weights'] * (1 - w_purchase)
        df_popularity_fin = df_X

        df_popularity_fin['popularity_score'] = (df_X['no_of_purchases'] * w_purchase + df_X['clicks'] * w_clicks + df_X['adds_to_cart_from_wishlist'] * w_wishlist + df_X['no_of_adds_to_cart'] * w_cart)/(w_purchase + w_clicks + w_wishlist + w_cart)
        df_popularity_fin.sort_values('popularity_score', ascending=False, inplace=True)
        df_popularity_fin.to_csv(f'{cfg.OUTPUT_DIR}/popularity_score.csv', index=False)

        return df_popularity_fin



############# Calling the Function #############
df_output = train_compare_models()
df_output_final = df_output.copy()


#### Uploading to Cassandra Table ####
# Connect to your local Cassandra instance
auth_provider = PlainTextAuthProvider(username=f'{cfg.USERNAME}', password=f'{cfg.PASSWORD}')
cluster = Cluster(contact_points=[f'{cfg.HOST_NAME}'], port=cfg.PORT, auth_provider=auth_provider)
session = cluster.connect()

# Create and use a keyspace
KEYSPACE_NAME = cfg.DB_NAME
session.execute(f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE_NAME} WITH REPLICATION = {{ 'class' : 'SimpleStrategy', 'replication_factor' : 1 }}")
session.set_keyspace(KEYSPACE_NAME)

# Create the Cassandra table
TABLE_NAME = cfg.TABLE_NAME

create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    brand text,
    clicks text,
    no_of_purchases text,
    no_of_unique_purchases text,
    adds_to_cart_from_wishlist text,
    no_of_adds_to_cart text,
    popularity_score text,
    PRIMARY KEY (brand)
)"""
session.execute(create_table_query)

# Insert DataFrame data into the Cassandra table
for index, row in df_output_final.iterrows():
    insert_query = f"""
    INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (brand, clicks, no_of_purchases, no_of_unique_purchases, adds_to_cart_from_wishlist, no_of_adds_to_cart,popularity_score) 
    VALUES ($${row['brand']}$$, '{row['clicks']}', '{row['no_of_purchases']}', '{row['no_of_unique_purchases']}','{row['adds_to_cart_from_wishlist']}', '{row['no_of_adds_to_cart']}', '{row['popularity_score']}')"""
    session.execute(insert_query)

print("\nTable uploaded successfully")

# Close the connection
cluster.shutdown()
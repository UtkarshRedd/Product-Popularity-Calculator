from pathlib import Path
import os

# # Setting Paths for directories
# class paths:

ROOT_DIR = Path(os.getcwd())
DATASETS_DIR = f'{ROOT_DIR}/Tables'
OUTPUT_DIR = f'{ROOT_DIR}/Output'
PLOTS_DIR = f'{ROOT_DIR}/Plots'
print(ROOT_DIR)
clicks_path = f"{ROOT_DIR}/Tables/prod_product_viewed.csv"
order_completed_path = f"{ROOT_DIR}/Tables/prod_order_completed.csv"
wishlist_path = f"{ROOT_DIR}/Tables/adds_to_cart_from_wishlist.csv"
cart_path = f"{ROOT_DIR}/Tables/prod_adds_to_cart.csv"
HOST_NAME = "127.0.0.1"
PORT = 9042
DB_NAME = 'prod_tira'
TABLE_NAME = 'popularity_table'
USERNAME = 'cassandra'
PASSWORD = 'cassandra'


## Actual cassandra cluster details
# HOST_NAME = "10.8.3.9"
# PORT = 9042
# DB_NAME = 'popularity_score'
# TABLE_NAME = 'tira_beauty_popularity'
# DB_USER='cassandra'
# DB_PASSWORD='4P3tVGEp27'

# # Local cassandra cluster details


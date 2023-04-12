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


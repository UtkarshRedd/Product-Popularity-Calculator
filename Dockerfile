# FROM "jioaicr.azurecr.io/jioaicr/python:3.9"
FROM python:3.9

ARG app_location='/usr/app'

WORKDIR ${app_location}

# Adding requirements
COPY requirements.txt $app_location/requirements.txt
COPY popularity_score.py $app_location/popularity_score.py
COPY config.py $app_location/config.py
COPY Tables/adds_to_cart_from_wishlist.csv $app_location/Tables/adds_to_cart_from_wishlist.csv
COPY Tables/prod_adds_to_cart.csv $app_location/Tables/prod_adds_to_cart.csv
COPY Tables/prod_adds_to_wishlist.csv $app_location/Tables/prod_adds_to_wishlist.csv
COPY Tables/prod_order_completed.csv $app_location/Tables/prod_order_completed.csv
COPY Tables/prod_product_viewed.csv $app_location/Tables/prod_product_viewed.csv
COPY Plots/RF_Feature_Importance.png $app_location/Plots/RF_Feature_Importance.png
COPY Plots/XGB_feature_importance.jpg $app_location/Plots/XGB_feature_importance.jpg
COPY Output/popularity_score.csv $app_location/Output/popularity_score.csv

# install dependencies
# ENV http_proxy "http://prodproxy.jio.com:8080"
# ENV https_proxy "http://prodproxy.jio.com:8080"
RUN pip install -r requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "popularity_score.py"]
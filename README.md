# Product-Popularity-Calculator
The main objective of this project is to design a Machine Learning pipeline that calculates the popularity of products from clickstream data obtained from the iOS, Android, and Web apps. This project had three main functionalities as shown in the diagram below: -

## Project Overview

![TIRA architecture](https://github.com/UtkarshRedd/Product-Popularity-Calculator/assets/29978378/2298a891-8c8c-4d88-9bba-14ae92438a8f)

Clickstream data was fetched from GCP BigQuery. The following click events were extracted from this data - Adds to cart, Adds to cart from wishlist, Number of clicks, Number of purchases, and Number of unique purchases. This project performs the following tasks: -

1. Analyzes click events – Number of clicks, Number of Adds to Cart, Number of Adds to Cart from Wishlist, Number of purchases.
2. Computes the influence/significance of these click events towards a unique product purchase and assigns respective weights to the events, using Regression.
3. Calculate the product popularity based on these weights.

The results of the significance of each click event, as illustrated by the best-performing regressor, are as follows: -


## Modules
There are a total of three modules in this project: -
1. Aggregate Popularity - Computes the popularity score of each product from the entire clickstream data (historical) starting from April 5, 2023.
2. Fortnightly/Monthly Popularity - Computes the popularity score of each product from the past two weeks or one month from clickstream data that starts from April 5, 2023.
3. Recency Bias - This feature enhances the popularity pipeline by incorporating a recency bias towards newly added products. The goal is to offset the naturally lower interaction levels that new items typically receive, preventing them from being overshadowed by older, more interacted-with products. It also adjusts for seasonal trends; for example, products that see a surge during festivals like Diwali or Holi won't dominate popularity scores indefinitely. With recency bias, older seasonal items will automatically receive lower weightage compared to newer products, regardless of their cumulative clicks or purchases.

## Relevant Files and Folders
1. The popularity_score.py contains the main machine learning pipeline that computes the product popularities without recency bias. It individually evaluates the performance of three regression models, namely Huber Regressor, XGBoost Regressor, and Random Forest Regressor, based on R2 score. It also outputs the feature importance plots of these models to the Plots folder.
2. The recency bias and fortnightly popularity pipelines are stored in the jupyter_notebooks folder. They haven't been productionized yet.
3. The Tables folder contains sample data.
4. recency_bias.py contains the main recency bias function.




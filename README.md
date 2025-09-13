# SCT_ML_02

# Customer Segmentation using K-Means Clustering

## 🛍️ Project Overview

This project implements a **K-means clustering algorithm** to segment customers of a retail store based on their purchase history. By grouping customers into distinct clusters, we can gain valuable insights into their behavior, preferences, and purchasing patterns. These insights can then be used to develop targeted marketing strategies, improve customer satisfaction, and drive business growth. This project utilizes the popular "Mall Customer" dataset to demonstrate the effectiveness of K-means clustering in a real-world retail scenario.

-----

## 📊 Dataset

The dataset used in this project is the **Mall Customer dataset**, which contains the following information about the customers:

  * **CustomerID**: A unique identifier for each customer.
  * **Gender**: The gender of the customer (Male/Female).
  * **Age**: The age of the customer.
  * **Annual Income (k$)**: The annual income of the customer in thousands of dollars.
  * **Spending Score (1-100)**: A score assigned to the customer based on their spending behavior and purchasing history.

-----

## 🤖 Methodology

The K-means clustering algorithm is an unsupervised machine learning algorithm that groups data points into a pre-defined number of clusters, denoted by 'K'. The algorithm works by iteratively assigning each data point to the nearest cluster centroid and then recalculating the centroid of each cluster. The process continues until the cluster assignments no longer change.

The steps to be followed in this project are:

1.  **Data Preprocessing**: The dataset is loaded, and any missing values or inconsistencies are handled.
2.  **Exploratory Data Analysis (EDA)**: The data is explored to understand the distribution of the features and the relationships between them. This includes creating visualizations such as histograms, scatter plots, and pair plots.
3.  **Optimal Number of Clusters**: The **Elbow Method** is used to determine the optimal number of clusters (K) for the K-means algorithm. This method involves plotting the within-cluster sum of squares (WCSS) for a range of K values and picking the "elbow" of the curve as the optimal K.
4.  **Model Training**: The K-means algorithm is trained on the selected features (in this case, 'Annual Income' and 'Spending Score') with the optimal number of clusters.
5.  **Cluster Visualization**: The resulting clusters are visualized using a scatter plot to understand the characteristics of each customer segment.

-----

## 📈 Results

The K-means clustering algorithm successfully grouped the customers into five distinct clusters:

  * **Cluster 1: High Income, Low Spending**: These are customers with high annual incomes but low spending scores. They are careful with their money and represent a potential opportunity for targeted marketing of high-value products.
  * **Cluster 2: Average Income, Average Spending**: This is the largest group of customers, with average annual incomes and spending scores. They are the "standard" customers.
  * **Cluster 3: Low Income, High Spending**: These customers have low annual incomes but high spending scores. They are likely to be younger customers who are more impulsive with their spending.
  * **Cluster 4: Low Income, Low Spending**: These customers have low annual incomes and low spending scores. They are cautious spenders and are likely to be a less profitable segment.
  * **Cluster 5: High Income, High Spending**: These are the ideal customers, with high annual incomes and high spending scores. They are the primary target for marketing campaigns and loyalty programs.

-----

## 🚀 How to Use

To run this project on your local machine, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook or Python script**:
    ```bash
    jupyter notebook customer_segmentation.ipynb
    ```
    or
    ```bash
    python customer_segmentation.py
    ```

-----

## 💻 Technologies Used

  * **Python**: The programming language used for this project.
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical computations.
  * **Matplotlib & Seaborn**: For data visualization.
  * **Scikit-learn**: For implementing the K-means clustering algorithm.

-----

## 🙏 Acknowledgments

  * The "Mall Customer" dataset is a popular dataset for practicing clustering algorithms.
  * Inspiration and guidance were drawn from various data science and machine learning resources.

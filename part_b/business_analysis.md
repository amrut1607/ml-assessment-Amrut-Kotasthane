# Part B: Business Case Analysis

## B1. Problem Formulation

### (a) Problem Formulation Details
- **Target Variable:** `items_sold` (a continuous numerical value representing the volume of items sold during a specific month/promotion period).
- **Candidate Input Features:** `store_size`, `monthly_footfall` (implicit proxy proxy variables or historical average), `local_competition_density`, `customer_demographics`, `location_type`, `promotion_type`, `is_weekend`, `is_festival`, and potentially temporal features like `month`.
- **Type of ML Problem:** This is a **Supervised Regression** problem, because the target variable (`items_sold`) is a continuous numerical outcome, and we have labeled historical data (features mappings to sales volume) to train the model.

**Justification:** While predicting *which* promotion to use could be framed as a classification or recommendation problem, predicting the continuous outcome (`items_sold`) for each store-promotion pair is more powerful. It allows the business to calculate the expected lift for every scenario and assign the promotion that yields the maximum projected `items_sold`.

### (b) Items Sold vs. Sales Revenue
Using `items_sold` (volume) is a much more reliable target variable here than total sales revenue because revenue is directly conflated with the promotion mechanic. For instance, a "Flat Discount" immediately reduces revenue per item, whereas a "Free Gift" might keep revenue intact but cost more on the backend. This creates biased metrics where revenue drops artificially simply because a price cut was introduced, masking the true measure of customer engagement and footfall conversion.

**Broader Principle:** In real-world ML projects, the target variable must be a clean, objective metric of the behavior you want to influence (i.e., demand generation), untethered from the mechanics of the treatment itself. Target variables should align directly with the untainted business objective.

### (c) Alternative Modelling Strategy
**Strategy: Clustered Models or Hierarchical Modeling (Mixed-Effects)**
Instead of a single global model (which might underfit local nuances) or 50 separate store models (which might suffer from insufficient data per store), a strong middle-ground is to segment stores using unsupervised clustering (e.g., K-Means on demographics, size, and location type) and build one model per cluster. Alternatively, tree-based models with store_id or cluster_id as a categorical feature, or mixed-effects models, can learn global patterns while retaining store-level intercepts.
**Justification:** This allows the model to leverage shared transactional data across similar geographical and demographic regions while explicitly accounting for structural differences, yielding better generalisation and robustness against sparse data for individual locations.

---

## B2. Data and EDA Strategy

### (a) Joining Tables & Final Grain
- **Join Strategy:**
  1. Start with the `transactions` table (the primary factual data).
  2. Left join the `calendar` table on `transaction_date` to pull weekend/festival flags.
  3. Left join the `promotion details` table on `date` and `store_id` (or whatever the promotion key is).
  4. Left join `store attributes` on `store_id`.
- **Final Grain:** One row = One specific **Store** in a specific **Month** with a specific **Promotion**. (If predictions are generated monthly, daily transactions must be aggregated to the monthly level).
- **Aggregations:** Sum `items_sold` and `revenue` per month. Calculate average daily footfall or total transaction counts per month. 

### (b) Exploratory Data Analysis (EDA)
1. **Target Distribution (`items_sold`):** A histogram or density plot to check for skewness or extreme outliers. *Influence:* If highly skewed right, I might apply a log-transformation to the target variable before modelling.
2. **Promotion vs. Items Sold (Boxplots):** Boxplots of `items_sold` split by `promotion_type`. *Influence:* Identifies visually if certain promotions vastly outperform others, and bounds expectations for feature importance.
3. **Correlation Heatmap:** Calculate correlation between numericals (like `store_size`, `competition_density`, `items_sold`). *Influence:* Helps to drop highly correlated overlapping features or guides the choice towards models resistant to multicollinearity (like Random Forests).
4. **Time-Series Plot (Sales over Months):** A line chart tracking aggregate volume over 3 years. *Influence:* Highlights seasonality. If heavy seasonality exists (e.g., Q4 spikes), engineering explicit seasonal features or lag variables becomes mandatory.

### (c) Addressing Class/Treatment Imbalance
Since 80% of historical transactions occur without any promotion, the model will see "No Promotion" as the dominant state. 
**Impact:** A model attempting to predict outcomes may under-represent the interaction effects of the rare promotion types, leading to poor predictions for those specific scenarios.
**Steps to Address:**
- Apply instance weighting (e.g., `sample_weight` in scikit-learn) to give more weight to rows where a promotion was active during training.
- Use tree-based models that partition the feature space cleanly based on the `promotion_type` dummy variables.
- Alternatively, frame the problem as an uplift model (predicting the *delta* in sales caused by the promotion).

---

## B3. Model Evaluation and Deployment

### (a) Setup Train-Test Split and Evaluation Metrics
- **Train-Test Split:** A **Temporal Split (Out-of-Time Validation)** setup is mandatory. The first 2.5 years of monthly data will serve as the training set, and the final 6 months as the hold-out test set. 
- **Why Random Split Fails:** A random shuffle leaks future target data and seasonal trends into the model's past, artificially inflating performance without simulating the real-world operational requirement of predicting the unknown future.
- **Evaluation Metrics:** 
  - **RMSE (Root Mean Squared Error):** Penalized heavily for large deviation errors on big stores. 
  - **MAE (Mean Absolute Error):** Provides an intuitive business interpretation (e.g., "On average, our prediction is off by X items").
  - **ROI/Lift (Business Metric):** Evaluating whether assigning the model's recommended promotion historically would yield higher `items_sold` than the baseline status quo.

### (b) Investigating Model Recommendations
Using the model's feature importance (like SHAP values or node impurity in Random forests), I would break down the predictions for Store 12 in December and March. 
I would communicate to marketing that the model calculates an expected `items_sold` based on interaction features. For December, SHAP values might reveal that the `is_festival` feature combined with the `Loyalty Points Bonus` produces the highest projected output because premium gift-buying during holidays heavily favors loyalty programs. In March (a standard, non-holiday month), SHAP would show that `is_festival` is false, pulling down Loyalty effectiveness, shifting the highest mathematical output to `Flat Discount` which acts as a more powerful generic volume-driver.

### (c) Deployment and Monitoring
1. **Saving the Model:** Serialize the scikit-learn pipeline (inclusive of all preprocessing and the trained regressor) using a robust library like `joblib` or `pickle` and store it in an artifact registry (e.g., AWS S3, MLflow).
2. **Scoring Pipeline:** Build a monthly batch prediction script. At the start of the month, the script will dynamically generate 5 rows per store (one for each possible `promotion_type` with current month features), feed all 250 rows (50 stores x 5 types) into the loaded model, and select the promotion with the highest predicted `items_sold` for each store.
3. **Monitoring:**
  - **Data Drift:** Monitor the distributions of incoming monthly features (e.g., if local competition suddenly spikes due to bad data).
  - **Model Degradation:** Calculate MAE and RMSE at the close of every month when true sales figures arrive. Set a threshold (e.g., "If MAE increases by >15% over 2 consecutive months, trigger automated retraining").

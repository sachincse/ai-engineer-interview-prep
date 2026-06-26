# Chapter 43 — Apparel Group AI/ML Mock Test (Timed Dry-Run)

> **Companion to [Chapter 42](42_apparel_group_ml_assessment.md).** This is a single, fresh, end-to-end mock that simulates the real Apparel Group online technical screen: **25 questions, ~40 minutes**, Mercer Mettl / iMocha auto-graded style. Questions are NEW (not copied from the Ch.42 bank). Every answer key was independently re-solved and audited (0 corrected, 0 replaced during QA).

## How to take it

1. **Set a 40-minute timer.** Don't pause it. ~1.5 min/question average; the coding question (Q25) will take longer, so move briskly through the MCQs.
2. **No scrolling to the answer key** until you've finished all 25. Write your letter answers (1→25) on paper.
3. **Simulate the real conditions:** full-screen, no other tabs, no notes, quiet room, laptop. The real test is likely webcam-proctored with tab-switch detection.
4. **Don't leave blanks** — negative marking is not stated in the invite; an educated guess beats a blank.
5. After time's up, score yourself against the key. **<16/25 → reread Ch.42 sections 4.1–4.6 before the real test.** 16–20 solid; 21+ you're ready.

---

## Part A — The Test (25 questions)

**Q1.** At 6thStreet, a model groups customers into clusters based on browsing behaviour without using any pre-assigned labels, so the marketing team can later inspect and name the segments. Which type of learning is this? _(Core ML fundamentals — supervised vs unsupervised · basic)_

- A. Supervised learning, because the output is customer segments
- B. Unsupervised learning, because the model finds structure with no target labels
- C. Reinforcement learning, because the marketing team rewards good segments
- D. Semi-supervised learning, because some customers have labels

**Q2.** A sales-forecasting model achieves 99% accuracy on the Apparel Group training data but only 71% on a held-out store's data. What is the most likely diagnosis? _(Core ML fundamentals — overfitting · basic)_

- A. Underfitting — the model is too simple
- B. Overfitting — the model memorized training noise and fails to generalize
- C. Data leakage from the test set into training
- D. The learning rate is too low

**Q3.** You replace a linear model with a depth-20 decision tree to predict daily SKU demand. Compared to the linear model, the tree most likely has: _(Core ML fundamentals — bias-variance · basic)_

- A. Higher bias and lower variance
- B. Lower bias and higher variance
- C. Lower bias and lower variance
- D. Higher bias and higher variance

**Q4.** Among 300 customer features, you want the model to automatically drop irrelevant ones by forcing some coefficients to be exactly zero. Which regularization should you use? _(Core ML fundamentals — regularization · basic)_

- A. L2 (Ridge) regularization
- B. L1 (Lasso) regularization
- C. Dropout
- D. Early stopping

**Q5.** Of 1,000 orders, only 20 are returns. A model that predicts 'no return' for every order reports 98% accuracy. Why is accuracy misleading here? _(Model evaluation — class imbalance · intermediate)_

- A. Accuracy is computed incorrectly; it should be 50%
- B. The classes are imbalanced, so high accuracy can be achieved while catching zero returns
- C. Accuracy always overstates recall
- D. The model is overfitting the majority class

**Q6.** A fraud classifier on 6thStreet orders gives this confusion matrix for the positive class 'fraud': TP=60, FP=20, FN=40, TN=880. What is the F1 score? _(Model evaluation — precision/recall/F1 calc · intermediate)_

- A. 0.60
- B. 0.6667
- C. 0.75
- D. 0.857

**Q7.** What does an ROC-AUC of 0.5 indicate for a binary return-prediction model? _(Model evaluation — ROC-AUC · intermediate)_

- A. Perfect ranking of positives above negatives
- B. The model ranks no better than random guessing
- C. The model has 50% accuracy
- D. The decision threshold is set to 0.5

**Q8.** You must predict the probability that a customer will redeem a discount coupon (yes/no). Why prefer logistic regression over linear regression? _(Classical algorithms — logistic vs linear regression · intermediate)_

- A. Logistic regression trains faster on large data
- B. Linear regression cannot use multiple features
- C. Logistic regression outputs values bounded in [0,1] via the sigmoid, suitable for probabilities
- D. Linear regression requires the target to be categorical

**Q9.** Which statement correctly distinguishes Random Forest from XGBoost? _(Classical algorithms — bagging vs boosting · intermediate)_

- A. Random Forest builds trees sequentially to correct prior errors; XGBoost builds them independently in parallel
- B. Random Forest builds independent trees in parallel (bagging); XGBoost builds trees sequentially, each correcting the residual errors of the previous (boosting)
- C. Both are bagging methods and differ only in tree depth
- D. XGBoost cannot handle tabular retail data; Random Forest can

**Q10.** When segmenting stores with K-means, why is it important to standardize features like annual_revenue (in millions) and num_employees (in tens) beforehand? _(Classical algorithms — K-means · intermediate)_

- A. K-means requires categorical features only
- B. Euclidean distance is dominated by large-magnitude features, so unscaled revenue would overwhelm the clustering
- C. Standardization guarantees the optimal number of clusters
- D. K-means cannot run on more than two features otherwise

**Q11.** You apply PCA to 50 correlated product-attribute features and keep the top 5 components. What do these 5 components represent? _(Classical algorithms — PCA · advanced)_

- A. The 5 original features with the highest variance
- B. 5 orthogonal linear combinations of the original features capturing the most variance
- C. The 5 features most correlated with the target label
- D. A random subset of 5 features chosen by cross-validation

**Q12.** What does this print? _(Python/NumPy — predict the output · intermediate)_

```python
import numpy as np
sales = np.array([[10, 20, 30], [40, 50, 60]])
print(sales.mean(axis=0))
```

- A. [20. 50.]
- B. [2.5 3.5 4.5]
- C. [25. 35. 45.]
- D. [35. 35. 35.]

**Q13.** When scaling features for a return-prediction model, which is the correct, leakage-free procedure? _(scikit-learn — fit/transform on train only · advanced)_

- A. Fit the StandardScaler on the full dataset, then split into train and test
- B. Fit the scaler on the training set only, then use transform on both train and test
- C. Fit and transform train and test independently with separate scalers
- D. Fit the scaler on the test set so it matches deployment data

**Q14.** What does this print? _(scikit-learn — train_test_split · intermediate)_

```python
from sklearn.model_selection import train_test_split
import numpy as np
X = np.arange(100).reshape(100, 1)
y = np.arange(100)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
print(Xtr.shape[0], Xte.shape[0])
```

- A. 100 20
- B. 80 20
- C. 20 80
- D. 50 50

**Q15.** Why wrap StandardScaler and LogisticRegression inside a single sklearn Pipeline before cross-validation? _(scikit-learn — Pipeline · intermediate)_

- A. It makes the model train faster on the GPU
- B. It ensures the scaler is re-fit on each CV training fold only, preventing leakage across folds
- C. It automatically tunes hyperparameters without GridSearchCV
- D. Pipelines are required to call .predict()

**Q16.** At 6thStreet, 2% of orders are fraudulent. The fraud model flags 95% of fraudulent orders (true positive) and falsely flags 5% of legitimate orders. Given an order is flagged, what is the probability it is actually fraud? (round to 2 decimals) _(Statistics — Bayes / conditional probability · advanced)_

- A. 0.95
- B. 0.28
- C. 0.50
- D. 0.02

**Q17.** An A/B test of a new checkout page returns p = 0.03 at significance level α = 0.05 for the hypothesis that conversion increased. What is the correct conclusion? _(Statistics — hypothesis testing / p-value · intermediate)_

- A. The new page has a 3% chance of being better
- B. Reject the null hypothesis; the observed lift is statistically significant at α = 0.05
- C. Accept the null hypothesis because p is small
- D. The effect size is 0.03

**Q18.** Two store features show a Pearson correlation of -0.85. What does this indicate? _(Statistics — correlation · intermediate)_

- A. A strong positive linear relationship
- B. A strong negative linear relationship
- C. No relationship at all
- D. A guaranteed causal relationship

**Q19.** For the output layer of a binary classifier predicting whether a customer churns, which activation function gives a valid probability? _(Deep learning — activation ranges · intermediate)_

- A. ReLU, which outputs [0, ∞)
- B. Sigmoid, which outputs (0, 1)
- C. Tanh, which outputs (-1, 1)
- D. Linear, which outputs (-∞, ∞)

**Q20.** You train a neural network to classify a product image into one of 12 mutually exclusive apparel categories. Which output activation and loss pairing is standard? _(Deep learning — loss choice · intermediate)_

- A. Sigmoid + binary cross-entropy
- B. Softmax + categorical cross-entropy
- C. Linear + mean squared error
- D. Tanh + hinge loss

**Q21.** A deep network overfits the Apparel Group sales data: training loss keeps dropping while validation loss rises. Which techniques can help REDUCE this overfitting? (Select all that apply.) _(Deep learning — dropout / overfitting (multi-answer) · advanced · select all that apply)_

- A. Add dropout layers
- B. Apply L2 weight regularization
- C. Increase the number of layers and neurons
- D. Use early stopping on validation loss

**Q22.** Apparel Group wants an LLM assistant that answers staff questions using a frequently-updated internal product catalog, without retraining the model each time the catalog changes. Which approach fits best? _(NLP/GenAI — RAG vs fine-tuning · advanced)_

- A. Fine-tune the LLM nightly on the full catalog
- B. Retrieval-Augmented Generation (RAG): embed catalog docs in a vector store and retrieve relevant chunks at query time
- C. Increase the model's temperature so it recalls the catalog
- D. Train a new model from scratch on the catalog

**Q23.** In an LLM that drafts product descriptions, setting temperature close to 0 will most likely: _(GenAI — temperature · intermediate)_

- A. Make outputs more random and creative
- B. Make outputs more deterministic and focused on the highest-probability tokens
- C. Increase the context window size
- D. Disable the attention mechanism

**Q24.** Given table sales(region, amount) with rows: ('Dubai',120),('Dubai',80),('Riyadh',200),('Dubai',50),('Riyadh',100). What does this return? _(SQL — GROUP BY · intermediate)_

```sql
SELECT region, AVG(amount) AS avg_amt
FROM sales
GROUP BY region
ORDER BY avg_amt DESC;
```

- A. Dubai 83.33, then Riyadh 150.00
- B. Riyadh 150.00, then Dubai 83.33
- C. Dubai 250, then Riyadh 300
- D. Riyadh 300, then Dubai 250

**Q25.** Implement mean_absolute_percentage_error(y_true, y_pred) WITHOUT using sklearn. It takes two equal-length lists of numbers (y_true has no zeros) and returns MAPE as a percentage: the mean over all samples of |(y_true - y_pred) / y_true|, multiplied by 100, as a float. Example: mean_absolute_percentage_error([100,200,300],[110,190,330]) should return 8.3333... . _(Python coding — implement metric from scratch · advanced)_

---

## Part B — Answer Key & Explanations

> Score 1 point per question (the multi-answer question requires all correct options for the point).

**Q1.** B. Unsupervised learning, because the model finds structure with no target labels

Clustering with no target variable is unsupervised; the algorithm discovers structure in unlabeled data. The team naming segments afterward does not constitute training labels.

**Q2.** B. Overfitting — the model memorized training noise and fails to generalize

A large train-test performance gap with high train accuracy is the classic signature of overfitting (high variance).

**Q3.** B. Lower bias and higher variance

A more flexible, high-capacity model reduces bias but increases variance, raising the risk of overfitting.

**Q4.** B. L1 (Lasso) regularization

L1/Lasso penalizes the sum of absolute coefficients and can drive some exactly to zero, performing feature selection; L2 only shrinks them toward (but not to) zero.

**Q5.** B. The classes are imbalanced, so high accuracy can be achieved while catching zero returns

With 98% negatives, a trivial majority-class predictor scores 98% accuracy yet has zero recall for returns — use precision/recall/F1 or AUC instead.

**Q6.** B. 0.6667

Precision = 60/(60+20) = 0.75, recall = 60/(60+40) = 0.60, so F1 = 2(0.75)(0.60)/(0.75+0.60) = 0.6667. Verified numerically.

**Q7.** B. The model ranks no better than random guessing

AUC is the probability a random positive is ranked above a random negative; 0.5 equals random, 1.0 is perfect, and AUC is threshold-independent (so option D is wrong).

**Q8.** C. Logistic regression outputs values bounded in [0,1] via the sigmoid, suitable for probabilities

Linear regression can produce values outside [0,1]; logistic regression's sigmoid maps the linear combination to a valid probability for classification.

**Q9.** B. Random Forest builds independent trees in parallel (bagging); XGBoost builds trees sequentially, each correcting the residual errors of the previous (boosting)

Bagging (Random Forest) averages independently trained trees to cut variance; boosting (XGBoost) fits trees sequentially on residuals/gradients to cut bias.

**Q10.** B. Euclidean distance is dominated by large-magnitude features, so unscaled revenue would overwhelm the clustering

K-means uses Euclidean distance, which is sensitive to feature scale; without standardization the high-magnitude feature dominates distance and skews clusters.

**Q11.** B. 5 orthogonal linear combinations of the original features capturing the most variance

PCA produces uncorrelated (orthogonal) principal components that are linear combinations of all original features, ordered by the variance they explain; it is unsupervised and ignores the target.

**Q12.** C. [25. 35. 45.]

axis=0 averages down each column: (10+40)/2=25, (20+50)/2=35, (30+60)/2=45. Verified by execution.

**Q13.** B. Fit the scaler on the training set only, then use transform on both train and test

Fitting on training data only and reusing those statistics to transform the test set prevents test-set information from leaking into preprocessing.

**Q14.** B. 80 20

test_size=0.2 of 100 rows yields 20 test rows and 80 train rows. Verified by execution.

**Q15.** B. It ensures the scaler is re-fit on each CV training fold only, preventing leakage across folds

A Pipeline re-fits preprocessing inside each fold's training data during cross-validation, so validation-fold statistics never leak into training.

**Q16.** B. 0.28

P(fraud|flag) = (0.95*0.02) / (0.95*0.02 + 0.05*0.98) = 0.019/0.068 ≈ 0.28; the low base rate keeps the posterior low despite high sensitivity. Verified numerically.

**Q17.** B. Reject the null hypothesis; the observed lift is statistically significant at α = 0.05

Since p (0.03) < α (0.05), we reject the null; the p-value is the probability of data this extreme under the null, not the probability the page is better.

**Q18.** B. A strong negative linear relationship

Pearson r near -1 indicates a strong negative linear association; correlation magnitude measures linear strength and does not imply causation.

**Q19.** B. Sigmoid, which outputs (0, 1)

Sigmoid squashes outputs to (0,1), interpretable as a probability for binary classification; ReLU, tanh, and linear are unbounded or signed.

**Q20.** B. Softmax + categorical cross-entropy

Multi-class single-label problems use softmax (a probability distribution over the 12 classes) with categorical cross-entropy loss.

**Q21.** A, B, D. Add dropout layers; Apply L2 weight regularization; Use early stopping on validation loss

Dropout, L2 regularization, and early stopping all reduce variance/overfitting; increasing capacity (more layers/neurons) generally worsens overfitting, so C is excluded.

**Q22.** B. Retrieval-Augmented Generation (RAG): embed catalog docs in a vector store and retrieve relevant chunks at query time

RAG injects fresh, authoritative context at inference time without retraining, ideal for frequently-changing knowledge bases and reducing hallucination.

**Q23.** B. Make outputs more deterministic and focused on the highest-probability tokens

Lower temperature sharpens the next-token distribution toward the most probable tokens, yielding more deterministic, less varied text.

**Q24.** B. Riyadh 150.00, then Dubai 83.33

Riyadh avg = (200+100)/2 = 150.00; Dubai avg = (120+80+50)/3 = 83.33; ORDER BY avg_amt DESC puts Riyadh first. Verified numerically.

**Q25.**

```python
def mean_absolute_percentage_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must be non-empty")
    total = 0.0
    for actual, pred in zip(y_true, y_pred):
        if actual == 0:
            raise ZeroDivisionError("y_true contains a zero, MAPE is undefined")
        total += abs((actual - pred) / actual)
    return (total / len(y_true)) * 100.0

# >>> mean_absolute_percentage_error([100, 200, 300], [110, 190, 330])
# 8.333333333333332
```

Sum the per-sample absolute percentage errors |(actual-pred)/actual|, divide by the number of samples, and multiply by 100. For the example: (0.10 + 0.05 + 0.10)/3 * 100 = 8.33%. Guards against length mismatch, empty input, and zero denominators. Confirmed runnable: returns 8.333333333333332.

---

_Compiled for Sachin Singh. Representative timed practice; not actual Apparel Group exam questions. See [Chapter 42](42_apparel_group_ml_assessment.md) for the full 163-question topic bank and format/company intel._
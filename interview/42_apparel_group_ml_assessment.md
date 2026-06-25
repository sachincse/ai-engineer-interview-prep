# Chapter 40 — Apparel Group AI/ML Engineer Online Assessment (Req #8682)

> **What this is.** Sachin was shortlisted for the **AI ML Engineer (8682)** role at **Apparel Group** (Dubai) and invited to an online **Technical** assessment — **under 45 minutes**, to be completed within **48 hours**. There is no public dump of Apparel Group’s exact test, so this chapter reconstructs the most probable content by cross-referencing the standard **Mercer Mettl / iMocha / HackerRank / Adaface / TestDome** “Machine Learning Engineer” screening templates and public interview banks, then **adversarially audits every answer key**. Treat these as *representative*, high-probability practice — not leaked questions.
>
> **How to drill it:** answers are hidden behind collapsible *Show answer* toggles so you can self-test. ⭐ marks the highest-yield questions. Skew your time toward sections 4.1–4.6.

**Bank size:** 163 verified questions across 10 sections (141 ⭐ high-yield; 0 draft questions dropped by the answer-key audit).

---

## 1. What this assessment is (format & logistics)

### What to Expect From This Assessment

You've been sent an automated technical screen by the Apparel Group Human Capital team, to be completed within 48 hours in roughly 45 minutes of uninterrupted, single-sitting time. Here's what that almost certainly means in practice.

**The platform.** For Gulf corporates running automated, no-reply, browser-locked technical screens, the dominant vendors are [Mercer | Mettl](https://mettl.com/en/test/machine-learning-engineer-assessment/) and [iMocha](https://www.imocha.io/tests/online-machine-learning-quiz-assessment-test). The strict "latest Edge/Chrome/Opera/Safari + stable internet + uninterrupted time" language is classic Mettl/iMocha boilerplate (their players need a modern browser for full-screen lock and webcam capture). Apparel Group is a Dubai-based group employer; either is plausible, with Mettl marginally more common in the GCC.

**Question count and pacing.** Mettl's standard [ML Engineer template is 19 MCQs in 60 minutes](https://mettl.com/en/test/machine-learning-engineer-assessment/); iMocha's mid-level ML test runs [15 questions in 20 minutes](https://www.imocha.io/tests/online-machine-learning-quiz-assessment-test). For a sub-45-minute test, expect roughly **18–25 items**, i.e. about **1.5–2.5 minutes per question** on average. Coding/pseudo-code items, if present, will eat more — budget for that.

**Question types.** Mostly auto-graded objective items: single-answer MCQs, multiple-answer (MAQ — read carefully, partial credit varies), fill-in-the-blank, and true/false. iMocha templates [also support](https://www.imocha.io/tests/online-machine-learning-quiz-assessment-test) one coding-simulator or **AI-LogicBox** pseudo-code item. Don't be surprised by a single hands-on coding/whiteboard question near the end.

**Section mix (typical ML Engineer template).** Three difficulty tiers — **basic / intermediate / advanced** ([per Mettl](https://mettl.com/en/test/machine-learning-engineer-assessment/)) — spread across core concepts (regression, classification, evaluation metrics, overfitting/regularization, train-test splits), advanced concepts (SVM, decision trees/ensembles, exploratory analysis, possibly deep learning), and hands-on Python/scikit-learn.

**Proctoring.** iMocha [explicitly offers](https://www.imocha.io/pre-employment-testing/machine-learning) webcam image/video capture, audio, and window-violation (tab-switch) detection; Mettl offers the same plus full-screen lock and copy-paste blocking. Assume you are being recorded and that leaving full-screen or switching tabs is logged. **Do not** open another tab, copy-paste, use a second monitor, talk aloud, or leave the camera frame.

**Practical test-taking tips:**

1. **Set up the environment first** — latest Chrome/Edge, charged/plugged-in laptop, wired or strong Wi-Fi, working webcam, good lighting, quiet room. Close all other apps and tabs *before* you start.
2. **Block the full window of time** — once started, the timer usually doesn't pause.
3. **Read MAQ stems twice** — "select all that apply" is where careless points are lost.
4. **Time-box hard items** — flag and move on; never sink 6 minutes into one question.
5. **Answer everything** — negative marking is *not* stated in your invite, but I cannot confirm it's absent; since it's unstated, a blank scores zero, so default to attempting all unless the test itself declares a penalty.
6. **Do the coding item last** if you can navigate freely — it's the highest time-risk.
7. **Don't tab-switch or copy-paste** — both are commonly flagged as violations.
8. **Stay in frame and on full-screen** for the whole session.
9. **Watch the on-screen timer**, and do a quick review pass only if time remains.
10. **Start with margin inside the 48-hour window** so a connection drop leaves room to contact support, not a missed deadline.

Sources: [Mercer Mettl ML Engineer test](https://mettl.com/en/test/machine-learning-engineer-assessment/), [iMocha ML test](https://www.imocha.io/tests/online-machine-learning-quiz-assessment-test), [iMocha ML test library](https://www.imocha.io/pre-employment-testing/machine-learning).

---

## 2. Apparel Group — company & role intel

### About Apparel Group

[Apparel Group](https://www.apparelgroup.com/en/all-brands/e-commerce/) is a Dubai/UAE-headquartered global fashion and lifestyle retail conglomerate operating 75+ brands across 2,000+ stores and employing roughly 12,000 people, primarily across the GCC and wider MENA/Asia region. It runs regional franchises and partnerships for names such as Nike, Tommy Hilfiger, Aldo, Charles & Keith, and Tim Hortons, and grew to roughly **$3.2 billion in revenue in 2024** ([appsruntheworld](https://www.appsruntheworld.com/customers-database/customers/view/apparel-group-llc-uae), [retail-insight-network](https://www.retail-insight-network.com/news/apparel-group-stores-middle-east/)). Its HR function is branded "Human Capital."

### Tech, Data & AI Context

Apparel Group runs its operations cloud-first and has publicly leaned into AI for retail optimization, from pricing to demand forecasting — leadership has gone as far as saying *"our AI is actually telling us whether a shoe should be sent to a store or not"* ([appsruntheworld](https://www.appsruntheworld.com/customers-database/customers/view/apparel-group-llc-uae)). The stack spans Adobe Analytics, Microsoft Dynamics NAV (ERP), Axonify, Cisco/Juniper/Citrix infrastructure, ISO/IEC 27001 security, and the Haptik conversational-AI platform. Active themes include **AI-powered forecasting, automated inventory, unified commerce, generative AI for content/personalization, loyalty programs**, and quick-commerce (60–90 minute delivery in Dubai, Abu Dhabi, Riyadh).

Its digital crown jewel is **[6thStreet](https://www.apparelgroup.com/en/all-brands/e-commerce/)**, the group's e-commerce arm carrying 1,200+ brands across the UAE, KSA, Kuwait, Bahrain, Qatar, and Oman. 6thStreet runs a mature personalization engine: using the MoEngage platform with "Sherpa" optimization and intelligent send-time delay, it reported **2.5x higher conversions and 3x higher CTRs** on automated vs. manual campaigns, plus a full browse→cart→purchase funnel and an omnichannel "phygital" store concept ([MoEngage case study](https://www.moengage.com/casestudy/6thstreet-com-uses-smart-recommendations-to-drive-higher-conversions/)).

### What an AI/ML Engineer Likely Does Day-to-Day

Inferring from a retail conglomerate of this scale, an AI/ML Engineer would likely build and productionize models against POS, e-commerce (6thStreet), CRM/loyalty, and supply-chain data on a cloud stack. Expect work on **demand forecasting and store-level allocation, recommendation/personalization systems, customer segmentation and churn, inventory and assortment optimization, pricing/markdown models, and increasingly GenAI** for product content, search, and chatbots (e.g., Haptik). The role probably blends classic ML (gradient-boosted trees, time-series) with MLOps — feature pipelines, model deployment/monitoring, and integrating outputs into merchandising and marketing tools ([apparelmagic](https://apparelmagic.com/the-role-of-ai-in-optimizing-supply-chain-management-in-the-fashion-industry/)).

### Likely Retail-AI Scenario Questions

- **Demand forecasting** — time-series/ML models for SKU-store-week demand, handling seasonality, promotions, weather, and new-product cold-start ([MDPI](https://www.mdpi.com/2571-9394/4/2/31)).
- **Inventory & allocation** — deciding stock distribution across stores, reducing stockouts and overstock/markdowns (research cites ~19% lower holding cost, ~24% fewer stockouts) ([ijsat](https://www.ijsat.org/papers/2025/1/2644.pdf)).
- **Recommendations & personalization** — collaborative filtering / embeddings for 6thStreet, ranking, and next-best-offer.
- **Customer churn & CLV / segmentation** — loyalty-program retention modeling and targeting.
- **Dynamic pricing & markdown optimization** — price elasticity and end-of-season clearance models.
- **Computer vision for apparel** — visual search, attribute tagging, virtual try-on, similar-item matching.
- **GenAI applications** — product description generation, conversational shopping assistants, and semantic search.

---

## 3. Section map

| § | Section | Qs | ⭐ |
|---|---------|----|----|
| 4.1 | Core ML Fundamentals | 16 | 14 |
| 4.2 | Model Evaluation & Metrics | 16 | 15 |
| 4.3 | Classical ML Algorithms | 19 | 17 |
| 4.4 | Python, NumPy, pandas & scikit-learn | 16 | 14 |
| 4.5 | Statistics & Probability | 16 | 13 |
| 4.6 | Deep Learning & Neural Networks | 16 | 15 |
| 4.7 | NLP, LLMs & Generative AI (2026) | 16 | 14 |
| 4.8 | SQL & Data Handling | 16 | 14 |
| 4.9 | Applied Retail / Apparel ML Scenarios | 16 | 13 |
| 4.10 | Hands-on Coding (Python) | 16 | 12 |

> The real test is short (≈18–25 items), so it will sample — not cover — all of this. The probability mass sits in **Core ML Fundamentals, Evaluation Metrics, Classical Algorithms, Python/sklearn, and Statistics**. Deep learning, NLP/LLM, SQL and coding appear in smaller amounts; retail scenarios are more likely in a later human interview than in the auto-graded screen.

---

## 4. Question bank

### 4.1 Core ML Fundamentals

> Supervised vs unsupervised vs reinforcement learning, bias-variance tradeoff, overfitting/underfitting, regularization (L1/L2/dropout/early stopping), train/val/test split, k-fold cross-validation, data leakage, feature scaling, when scaling is needed.
>
> **16 questions**, 14 ⭐ high-yield.

**Q1.** Apparel Group wants to group thousands of customers into purchase-behaviour segments, but there are no predefined segment labels in the data. Which type of machine learning is most appropriate?  ⭐ _(basic)_

- A. Supervised learning
- B. Unsupervised learning
- C. Reinforcement learning
- D. Semi-supervised regression

<details>
<summary>Show answer</summary>

**B. Unsupervised learning**

Clustering customers when no target/label exists is unsupervised learning (e.g., k-means). Supervised learning requires labelled targets, and reinforcement learning needs an agent learning from reward signals.

</details>

**Q2.** Which of the following is a SUPERVISED learning task?  ⭐ _(basic)_

- A. Grouping similar products into clusters with no labels
- B. Reducing 100 features to 2 dimensions with PCA
- C. Predicting next month's sales revenue from historical labelled data
- D. Learning a warehouse-robot policy through trial-and-error rewards

<details>
<summary>Show answer</summary>

**C. Predicting next month's sales revenue from historical labelled data**

Predicting a known numeric target (sales) from labelled examples is supervised regression. Clustering and PCA are unsupervised; learning via reward feedback is reinforcement learning.

</details>

**Q3.** A model achieves 99% accuracy on the training set but only 71% on the test set. What is the most likely problem?  ⭐ _(basic)_

- A. Underfitting (high bias)
- B. Overfitting (high variance)
- C. Data is perfectly clean
- D. The learning rate is too high

<details>
<summary>Show answer</summary>

**B. Overfitting (high variance)**

A large gap with high training accuracy and much lower test accuracy is the classic signature of overfitting/high variance. Underfitting would show poor performance on BOTH train and test.

</details>

**Q4.** A model has both high training error and high test error, with the two errors close together. This indicates:  ⭐ _(intermediate)_

- A. High variance / overfitting
- B. High bias / underfitting
- C. Data leakage
- D. Perfect generalization

<details>
<summary>Show answer</summary>

**B. High bias / underfitting**

Poor performance on both training and test sets (errors high and similar) is underfitting, caused by a model too simple to capture the underlying pattern (high bias). Overfitting instead shows low train error but high test error.

</details>

**Q5.** Regarding the bias-variance tradeoff, which statement is correct?  ⭐ _(intermediate)_

- A. Increasing model complexity typically decreases bias but increases variance
- B. Increasing model complexity decreases both bias and variance simultaneously
- C. Variance measures the model's systematic error from wrong assumptions
- D. A high-bias model is highly sensitive to small changes in the training data

<details>
<summary>Show answer</summary>

**A. Increasing model complexity typically decreases bias but increases variance**

More complex models fit training data more closely (lower bias) but become more sensitive to the specific training sample (higher variance). Bias is the systematic error from oversimplified assumptions; variance is the sensitivity to training-data fluctuations.

</details>

**Q6.** Which regularization technique can drive some feature coefficients to be exactly zero, effectively performing feature selection?  ⭐ _(intermediate)_

- A. L2 (Ridge) regularization
- B. L1 (Lasso) regularization
- C. Dropout
- D. Batch normalization

<details>
<summary>Show answer</summary>

**B. L1 (Lasso) regularization**

L1 (Lasso) adds the sum of absolute values of coefficients as a penalty, which shrinks some coefficients exactly to zero, yielding sparse models and built-in feature selection. L2 (Ridge) only shrinks coefficients toward (but not exactly to) zero.

</details>

**Q7.** L2 (Ridge) regularization adds which penalty term to the loss function?  ⭐ _(intermediate)_

- A. The sum of the absolute values of the weights
- B. The sum of the squared values of the weights
- C. The count of non-zero weights
- D. The maximum absolute weight value

<details>
<summary>Show answer</summary>

**B. The sum of the squared values of the weights**

L2 / Ridge penalizes the squared magnitude of the weights (lambda times the sum of w^2), shrinking them smoothly toward zero. The sum of absolute values is the L1 penalty.

</details>

**Q8.** In a neural network, dropout reduces overfitting by:  ⭐ _(intermediate)_

- A. Permanently deleting unimportant neurons after training
- B. Randomly deactivating a fraction of neurons during each training step
- C. Adding the squared weights to the loss function
- D. Increasing the learning rate over time

<details>
<summary>Show answer</summary>

**B. Randomly deactivating a fraction of neurons during each training step**

Dropout randomly sets a fraction of activations to zero during each training iteration, preventing co-adaptation and forcing redundancy, which improves generalization. Neurons are not permanently removed; at inference all neurons are used (scaled).

</details>

**Q9.** Early stopping prevents overfitting by:  ⭐ _(intermediate)_

- A. Stopping training when the validation error starts to increase
- B. Stopping training as soon as training error reaches zero
- C. Removing outliers from the training set before training
- D. Reducing the number of input features

<details>
<summary>Show answer</summary>

**A. Stopping training when the validation error starts to increase**

Early stopping monitors validation error and halts training once it begins to rise (while training error keeps falling), capturing the model before it overfits. Stopping at zero training error would typically be deep into overfitting.

</details>

**Q10.** What is the primary purpose of the VALIDATION set (as distinct from the test set)?  ⭐ _(intermediate)_

- A. To train the model's weights
- B. To tune hyperparameters and select models during development
- C. To provide the final unbiased estimate of generalization performance
- D. To increase the total amount of training data

<details>
<summary>Show answer</summary>

**B. To tune hyperparameters and select models during development**

The validation set is used to tune hyperparameters and choose between models. The test set is held out and used only once at the end for an unbiased generalization estimate; the training set fits the weights.

</details>

**Q11.** In k-fold cross-validation with k = 5, how is the data used?  ⭐ _(intermediate)_

- A. The data is split into 5 folds; the model trains on 4 folds and validates on the remaining 1, repeated 5 times
- B. The model is trained 5 times on the entire dataset and the results are averaged
- C. The data is split once into 5% test and 95% train
- D. 5 different models are trained on 5 completely separate datasets

<details>
<summary>Show answer</summary>

**A. The data is split into 5 folds; the model trains on 4 folds and validates on the remaining 1, repeated 5 times**

k-fold CV partitions the data into k equal folds; each fold serves once as the validation set while the other k-1 train the model, and the k scores are averaged for a more robust performance estimate.

</details>

**Q12.** You standardize (fit the scaler on) the ENTIRE dataset and then split it into train and test sets. What problem does this cause?  ⭐ _(advanced)_

- A. Underfitting, because scaling reduces model capacity
- B. Data leakage, because test-set statistics influence the training transformation
- C. Nothing wrong; this is the recommended approach
- D. Class imbalance in the test set

<details>
<summary>Show answer</summary>

**B. Data leakage, because test-set statistics influence the training transformation**

Fitting the scaler on all data lets the test set's mean/variance leak into preprocessing, giving optimistically biased test scores. The correct practice is to fit the scaler on the training data only and apply it to the test set.

</details>

**Q13.** Which scenario is an example of TARGET (label) leakage?  ⭐ _(advanced)_

- A. Using last month's sales to predict this month's sales
- B. Including a 'refund_issued' flag as a feature to predict whether a purchase will be returned
- C. Scaling features after the train/test split
- D. Using one-hot encoding for a categorical product-category feature

<details>
<summary>Show answer</summary>

**B. Including a 'refund_issued' flag as a feature to predict whether a purchase will be returned**

A 'refund_issued' flag is only known after the return happens, so it leaks the outcome into the features and inflates performance unrealistically. Lagged features and post-split scaling are legitimate practices.

</details>

**Q14.** Which group of algorithms is generally SENSITIVE to feature scaling (and therefore benefits from standardization/normalization)?  ⭐ _(intermediate)_

- A. Decision Tree, Random Forest, Gradient-Boosted Trees
- B. KNN, SVM, and gradient-descent-based models (e.g., logistic/linear regression, neural nets)
- C. Naive Bayes only
- D. All tree-based ensembles

<details>
<summary>Show answer</summary>

**B. KNN, SVM, and gradient-descent-based models (e.g., logistic/linear regression, neural nets)**

Distance-based (KNN, SVM, k-means) and gradient-descent-trained models are sensitive to feature scale, so scaling helps. Tree-based methods (options A and D) split on per-feature thresholds and are invariant to monotonic rescaling, so only option B is correct.

</details>

**Q15.** An online warehouse robot at Apparel Group learns to pick and pack items by receiving a reward for each correctly fulfilled order and a penalty for errors, improving its policy over time. This is best described as: _(basic)_

- A. Supervised learning
- B. Unsupervised learning
- C. Reinforcement learning
- D. Dimensionality reduction

<details>
<summary>Show answer</summary>

**C. Reinforcement learning**

An agent learning a policy through trial-and-error using reward/penalty feedback from the environment is reinforcement learning. There are no fixed input-output labels (supervised) and the goal is not to find hidden structure (unsupervised).

</details>

**Q16.** Select ALL statements that are TRUE about regularization. (Select all that apply.) _(advanced)_

- A. Regularization adds a penalty term to the loss to discourage overly complex models
- B. Increasing the regularization strength (λ) generally increases bias and decreases variance
- C. L1 regularization tends to produce sparse weight vectors
- D. Regularization is used primarily to fix underfitting (high bias)

<details>
<summary>Show answer</summary>

**A, B, C. Regularization adds a penalty term to the loss to discourage overly complex models; Increasing the regularization strength (λ) generally increases bias and decreases variance; L1 regularization tends to produce sparse weight vectors**

Regularization penalizes complexity, stronger lambda raises bias while lowering variance, and L1 yields sparsity. It is used to combat OVERfitting (high variance), not underfitting, so the last statement is false.

</details>

### 4.2 Model Evaluation & Metrics

> Accuracy/precision/recall/F1 (and exact formulas), confusion matrix, ROC-AUC vs PR curve, MSE/RMSE/MAE/R-squared, log loss, handling class imbalance, choosing a metric for a given problem.
>
> **16 questions**, 15 ⭐ high-yield.

**Q1.** What is the correct formula for Precision in a binary classification problem?  ⭐ _(basic)_

- A. TP / (TP + FN)
- B. TP / (TP + FP)
- C. (TP + TN) / (TP + TN + FP + FN)
- D. TN / (TN + FP)

<details>
<summary>Show answer</summary>

**B. TP / (TP + FP)**

Precision answers 'of all items predicted positive, how many were truly positive', so it divides true positives by all predicted positives (TP + FP). TP/(TP+FN) is recall, and TN/(TN+FP) is specificity.

</details>

**Q2.** Recall (sensitivity / true positive rate) is defined as:  ⭐ _(basic)_

- A. TP / (TP + FP)
- B. TP / (TP + FN)
- C. FP / (FP + TN)
- D. TN / (TN + FN)

<details>
<summary>Show answer</summary>

**B. TP / (TP + FN)**

Recall measures the fraction of actual positives correctly identified, dividing true positives by all real positives (TP + FN). TP/(TP+FP) is precision, FP/(FP+TN) is the false positive rate.

</details>

**Q3.** The F1 score is best described as the:  ⭐ _(basic)_

- A. Arithmetic mean of precision and recall
- B. Harmonic mean of precision and recall
- C. Geometric mean of accuracy and recall
- D. Product of precision and recall

<details>
<summary>Show answer</summary>

**B. Harmonic mean of precision and recall**

F1 = 2 x (Precision x Recall) / (Precision + Recall), the harmonic mean of precision and recall. The harmonic mean is used because it is dominated by the smaller of the two values, so a high F1 requires both precision and recall to be high.

</details>

**Q4.** A garment-defect classifier is evaluated on 1,000 items: TP = 80, FP = 20, FN = 40, TN = 860. What is its precision?  ⭐ _(intermediate)_

- A. 0.667
- B. 0.80
- C. 0.94
- D. 0.727

<details>
<summary>Show answer</summary>

**B. 0.80**

Precision = TP / (TP + FP) = 80 / (80 + 20) = 80/100 = 0.80. The 0.667 distractor is recall (80/120) and 0.94 is overall accuracy ((80+860)/1000).

</details>

**Q5.** Using the same confusion matrix (TP = 80, FP = 20, FN = 40, TN = 860), what is the recall?  ⭐ _(intermediate)_

- A. 0.80
- B. 0.94
- C. 0.667
- D. 0.50

<details>
<summary>Show answer</summary>

**C. 0.667**

Recall = TP / (TP + FN) = 80 / (80 + 40) = 80/120 ≈ 0.667. The 0.80 distractor is precision (80/100).

</details>

**Q6.** For an online apparel store, the fraud-detection model flags only 0.5% of transactions as fraudulent. Why is plain accuracy a poor metric here?  ⭐ _(intermediate)_

- A. Accuracy cannot be computed when classes are imbalanced
- B. A model that predicts 'not fraud' for every transaction would score ~99.5% accuracy while catching zero fraud
- C. Accuracy requires probability outputs, which fraud models do not produce
- D. Accuracy always equals recall in imbalanced datasets

<details>
<summary>Show answer</summary>

**B. A model that predicts 'not fraud' for every transaction would score ~99.5% accuracy while catching zero fraud**

With severe class imbalance, always predicting the majority class yields very high accuracy yet is useless for the minority class. Precision, recall, F1, or PR-AUC reflect minority-class performance better.

</details>

**Q7.** In a confusion matrix, a False Negative (FN) occurs when:  ⭐ _(basic)_

- A. The model predicts positive and the true label is positive
- B. The model predicts negative but the true label is positive
- C. The model predicts positive but the true label is negative
- D. The model predicts negative and the true label is negative

<details>
<summary>Show answer</summary>

**B. The model predicts negative but the true label is positive**

A false negative is a real positive that the model missed (predicted negative). Predicting positive on a true negative is a false positive; predicting negative on a true negative is a true negative.

</details>

**Q8.** For a highly imbalanced dataset (e.g., 1% positive class), which curve is generally MORE informative than the ROC curve for evaluating a binary classifier?  ⭐ _(advanced)_

- A. The Precision-Recall (PR) curve
- B. The ROC curve is always best regardless of imbalance
- C. The calibration (reliability) curve
- D. The learning curve

<details>
<summary>Show answer</summary>

**A. The Precision-Recall (PR) curve**

ROC uses the false positive rate, which stays low when true negatives are abundant, making ROC-AUC look deceptively optimistic under heavy imbalance. PR curves use precision and recall, which focus on the positive class (TP, FP, FN) and expose poor minority-class performance.

</details>

**Q9.** An ROC-AUC value of exactly 0.5 indicates that the classifier:  ⭐ _(intermediate)_

- A. Is a perfect classifier
- B. Performs no better than random guessing
- C. Is perfectly miscalibrated but accurate
- D. Has 50% accuracy

<details>
<summary>Show answer</summary>

**B. Performs no better than random guessing**

AUC = 1.0 is perfect ranking, 0.5 is equivalent to random guessing, and below 0.5 means worse than random (inverted ranking). AUC measures ranking/discrimination ability, not raw accuracy, so option D conflating it with 50% accuracy is wrong.

</details>

**Q10.** A demand-forecasting regression model predicts weekly unit sales. Compared to MAE, why is RMSE more sensitive to a few weeks with very large prediction errors?  ⭐ _(intermediate)_

- A. RMSE ignores the sign of errors while MAE does not
- B. RMSE squares the errors before averaging, so large errors are weighted more heavily
- C. RMSE divides by the variance of the target
- D. RMSE uses absolute values, which amplify outliers

<details>
<summary>Show answer</summary>

**B. RMSE squares the errors before averaging, so large errors are weighted more heavily**

RMSE = sqrt(mean of squared errors); squaring inflates large residuals disproportionately, making RMSE penalize big misses (outliers) more than MAE, which averages absolute errors linearly. Both metrics ignore sign, so option A does not distinguish them.

</details>

**Q11.** What does an R-squared (coefficient of determination) value of 0.85 indicate for a regression model?  ⭐ _(intermediate)_

- A. The model's predictions are off by 85% on average
- B. 85% of the variance in the target variable is explained by the model
- C. The model has 85% classification accuracy
- D. The correlation between any two features is 0.85

<details>
<summary>Show answer</summary>

**B. 85% of the variance in the target variable is explained by the model**

R-squared = 1 - SS_res/SS_tot measures the proportion of variance in the target explained by the model. It is unitless and unrelated to classification accuracy or average percentage error.

</details>

**Q12.** A spam filter for customer-service emails must avoid wrongly sending legitimate emails to spam (a false positive). Which metric should be prioritized?  ⭐ _(intermediate)_

- A. Recall
- B. Precision
- C. Mean Absolute Error
- D. Specificity of the negative class only

<details>
<summary>Show answer</summary>

**B. Precision**

Treating 'spam' as the positive class, a false positive is a legit email marked spam. Precision = TP/(TP+FP) directly penalizes false positives, so maximizing it minimizes the costly error. Recall would be prioritized when missing positives (letting spam through) is the costly error.

</details>

**Q13.** Log loss (binary cross-entropy) primarily rewards a classifier for:  ⭐ _(advanced)_

- A. Producing well-calibrated, confident probabilities that match the true labels
- B. Maximizing the number of correct hard predictions only
- C. Having a balanced confusion matrix
- D. Minimizing the number of features used

<details>
<summary>Show answer</summary>

**A. Producing well-calibrated, confident probabilities that match the true labels**

Log loss penalizes the predicted probability assigned to the true class; a confident wrong prediction (e.g., p=0.99 for the wrong class) is punished heavily, while confident correct predictions are rewarded. Unlike accuracy, it evaluates the probability estimates, not just the thresholded label.

</details>

**Q14.** A model has precision = 0.6 and recall = 0.4. What is its F1 score? _(intermediate)_

- A. 0.50
- B. 0.24
- C. 0.48
- D. 0.20

<details>
<summary>Show answer</summary>

**C. 0.48**

F1 = 2 x (0.6 x 0.4) / (0.6 + 0.4) = 2 x 0.24 / 1.0 = 0.48. Note it is below the arithmetic mean (0.50), since the harmonic mean is pulled toward the smaller value. The 0.24 distractor is the product P x R.

</details>

**Q15.** Lowering the classification probability threshold (e.g., from 0.5 to 0.3) for the positive class generally has which effect?  ⭐ _(advanced)_

- A. Increases recall but tends to decrease precision
- B. Increases both precision and recall equally
- C. Decreases recall but increases precision
- D. Has no effect on precision or recall

<details>
<summary>Show answer</summary>

**A. Increases recall but tends to decrease precision**

A lower threshold labels more instances positive, so more true positives are caught (recall up, monotonically) but more false positives also appear (precision typically down). This is the classic precision-recall trade-off.

</details>

**Q16.** Which metrics are appropriate choices for evaluating a REGRESSION model? (Select all that apply.)  ⭐ _(basic)_

- A. Root Mean Squared Error (RMSE)
- B. R-squared
- C. F1 score
- D. Mean Absolute Error (MAE)

<details>
<summary>Show answer</summary>

**A, B, D. RMSE; R-squared; Mean Absolute Error (MAE)**

RMSE, R-squared, and MAE all measure continuous-output error or goodness-of-fit and apply to regression. F1 score is a classification metric (harmonic mean of precision and recall) and does not apply to regression.

</details>

### 4.3 Classical ML Algorithms

> Linear & logistic regression, decision trees, random forest, bagging vs boosting, XGBoost/GBM, SVM (kernels, margin), KNN, K-means, Naive Bayes, PCA, gradient descent (batch/stochastic/mini-batch), learning rate.
>
> **19 questions**, 17 ⭐ high-yield.

**Q1.** In logistic regression, what function maps the linear combination of inputs to a probability between 0 and 1?  ⭐ _(basic)_

- A. Sigmoid (logistic) function
- B. ReLU function
- C. Identity (linear) function
- D. Step function

<details>
<summary>Show answer</summary>

**A. Sigmoid (logistic) function**

Logistic regression applies the sigmoid 1/(1+e^-z) to the linear score z, squashing it into (0,1) so it can be read as a probability. ReLU and identity are unbounded; a hard step gives no gradient for training.

</details>

**Q2.** An apparel retailer wants to predict whether a customer will churn (yes/no) from their loyalty program. Which algorithm directly outputs a class probability for this binary target?  ⭐ _(basic)_

- A. Linear regression
- B. Logistic regression
- C. K-means clustering
- D. PCA

<details>
<summary>Show answer</summary>

**B. Logistic regression**

Logistic regression is built for binary classification and outputs P(class=1). Linear regression predicts unbounded continuous values, K-means is unsupervised clustering, and PCA is dimensionality reduction, so none directly gives a churn probability.

</details>

**Q3.** Which criterion do classification decision trees (e.g., CART) commonly use to decide the best split at a node?  ⭐ _(basic)_

- A. Mean squared error only
- B. Gini impurity or entropy (information gain)
- C. Euclidean distance to centroids
- D. Cosine similarity

<details>
<summary>Show answer</summary>

**B. Gini impurity or entropy (information gain)**

Classification trees pick the split that most reduces impurity, measured by Gini impurity or entropy/information gain. (Strictly, CART itself uses Gini; entropy/information gain is the ID3/C4.5 criterion — both are standard classification-tree impurity measures.) MSE is used for regression trees, and distance/similarity measures belong to KNN and clustering, not tree splitting.

</details>

**Q4.** A single deep decision tree achieves 100% training accuracy but performs poorly on the test set. This is a classic symptom of:  ⭐ _(basic)_

- A. High bias (underfitting)
- B. Overfitting (high variance)
- C. Vanishing gradients
- D. Class imbalance

<details>
<summary>Show answer</summary>

**B. Overfitting (high variance)**

Perfect training accuracy with poor generalization means the unpruned tree memorized the training data — high variance / overfitting. Pruning, limiting depth, or using an ensemble like Random Forest reduces it.

</details>

**Q5.** What is the KEY difference between bagging and boosting?  ⭐ _(intermediate)_

- A. Bagging trains base learners independently in parallel; boosting trains them sequentially, each correcting the previous one's errors
- B. Bagging is sequential while boosting is parallel
- C. Bagging can only use decision trees while boosting can use any model
- D. Bagging increases bias while boosting increases variance

<details>
<summary>Show answer</summary>

**A. Bagging trains base learners independently in parallel; boosting trains them sequentially, each correcting the previous one's errors**

Bagging (e.g., Random Forest) trains independent models on bootstrap samples and averages/votes, mainly reducing variance. Boosting builds learners sequentially, each weighting/fitting the residual errors of the prior, mainly reducing bias.

</details>

**Q6.** In a Random Forest, what two sources of randomness make the individual trees less correlated?  ⭐ _(intermediate)_

- A. Bootstrap sampling of rows AND random feature subset selection at each split
- B. Random learning rate AND random tree depth
- C. Random initial centroids AND random K
- D. Shuffling labels AND adding Gaussian noise to features

<details>
<summary>Show answer</summary>

**A. Bootstrap sampling of rows AND random feature subset selection at each split**

Random Forest combines bagging (bootstrap row samples per tree) with random feature subsetting at each split. Decorrelating the trees this way reduces variance more than bagging alone.

</details>

**Q7.** Which of the following are boosting algorithms? (Select all that apply.)  ⭐ _(intermediate)_

- A. AdaBoost
- B. Gradient Boosting Machine (GBM)
- C. XGBoost
- D. Random Forest

<details>
<summary>Show answer</summary>

**A, B, C. AdaBoost, Gradient Boosting Machine (GBM), XGBoost**

AdaBoost, GBM, and XGBoost all build models sequentially to correct prior errors (boosting). Random Forest is a bagging method that trains trees independently, so it is not a boosting algorithm.

</details>

**Q8.** Why does XGBoost often outperform a plain Gradient Boosting Machine in practice?  ⭐ _(intermediate)_

- A. It uses unsupervised pretraining
- B. It adds regularization (L1/L2) plus engineering optimizations like parallelized split-finding and handling of sparse/missing data
- C. It removes the need for a learning rate
- D. It replaces trees with linear models

<details>
<summary>Show answer</summary>

**B. It adds regularization (L1/L2) plus engineering optimizations like parallelized split-finding and handling of sparse/missing data**

XGBoost is regularized gradient boosting: it adds L1/L2 penalties on leaf weights to curb overfitting and uses cache-aware, parallelized split-finding and built-in missing-value handling for speed and scalability. It still uses trees and a learning rate (shrinkage).

</details>

**Q9.** In a Support Vector Machine, what does the 'margin' refer to?  ⭐ _(intermediate)_

- A. The misclassification rate on the training set
- B. The distance between the decision boundary (hyperplane) and the nearest data points (support vectors)
- C. The number of support vectors used
- D. The learning rate of the optimizer

<details>
<summary>Show answer</summary>

**B. The distance between the decision boundary (hyperplane) and the nearest data points (support vectors)**

An SVM finds the hyperplane that maximizes the margin — the gap to the closest points (support vectors) of each class. A wider margin generally gives better generalization.

</details>

**Q10.** An SVM must separate classes that are NOT linearly separable in the original feature space. What is the standard SVM technique to handle this?  ⭐ _(intermediate)_

- A. Increase the learning rate
- B. Use a non-linear kernel (e.g., RBF/Gaussian) to implicitly map data to a higher-dimensional space
- C. Remove the support vectors
- D. Switch the loss to mean squared error

<details>
<summary>Show answer</summary>

**B. Use a non-linear kernel (e.g., RBF/Gaussian) to implicitly map data to a higher-dimensional space**

The kernel trick lets the SVM compute inner products in a higher-dimensional space without explicit mapping, making non-linearly-separable data separable. RBF is the most common non-linear kernel.

</details>

**Q11.** In the soft-margin SVM, what does a LARGER value of the regularization parameter C do? _(advanced)_

- A. Penalizes margin violations more heavily, yielding a narrower margin and risking overfitting
- B. Always widens the margin and ignores misclassifications
- C. Has no effect on a linear kernel
- D. Reduces the number of features

<details>
<summary>Show answer</summary>

**A. Penalizes margin violations more heavily, yielding a narrower margin and risking overfitting**

C controls the trade-off between maximizing the margin and minimizing slack (misclassifications). Large C = low tolerance for errors -> narrower margin, higher variance; small C = wider, softer margin with more tolerated violations.

</details>

**Q12.** KNN is described as a 'lazy learner.' What does that mean?  ⭐ _(intermediate)_

- A. It builds an explicit model during a long training phase
- B. It does almost no work at training time and defers computation (distance calculations) to prediction time
- C. It can only be used for clustering
- D. It ignores the value of K

<details>
<summary>Show answer</summary>

**B. It does almost no work at training time and defers computation (distance calculations) to prediction time**

KNN simply stores the training data; at prediction it computes distances to find the K nearest neighbors and votes/averages. This makes training trivial but inference expensive on large datasets.

</details>

**Q13.** Before applying KNN or K-means, why is feature scaling (normalization/standardization) usually important?  ⭐ _(intermediate)_

- A. Because both rely on distance metrics, so features on larger numeric scales would otherwise dominate the distance
- B. Because it makes the labels balanced
- C. Because it reduces the number of clusters
- D. Because it converts the problem to regression

<details>
<summary>Show answer</summary>

**A. Because both rely on distance metrics, so features on larger numeric scales would otherwise dominate the distance**

KNN and K-means use Euclidean (or similar) distances; an unscaled large-range feature (e.g., price) would swamp small-range features (e.g., rating). Scaling puts features on comparable ranges so distance is meaningful.

</details>

**Q14.** What does the parameter K represent in the K-means algorithm?  ⭐ _(basic)_

- A. The number of nearest neighbors
- B. The number of clusters to form
- C. The number of features
- D. The maximum number of iterations

<details>
<summary>Show answer</summary>

**B. The number of clusters to form**

In K-means, K is the predefined number of clusters (centroids) the algorithm partitions the data into. Note this differs from KNN, where K is the number of neighbors.

</details>

**Q15.** The Naive Bayes classifier is called 'naive' because it assumes:  ⭐ _(basic)_

- A. Features are conditionally independent given the class label
- B. The data is linearly separable
- C. All classes are equally probable
- D. There is no noise in the data

<details>
<summary>Show answer</summary>

**A. Features are conditionally independent given the class label**

Naive Bayes assumes each feature contributes independently to the class probability given the label — an unrealistic but simplifying 'naive' assumption that still works surprisingly well (e.g., text/spam classification).

</details>

**Q16.** What is the primary purpose of Principal Component Analysis (PCA)?  ⭐ _(intermediate)_

- A. Supervised classification of labeled data
- B. Dimensionality reduction by projecting data onto directions of maximum variance
- C. Increasing the number of features
- D. Balancing imbalanced classes

<details>
<summary>Show answer</summary>

**B. Dimensionality reduction by projecting data onto directions of maximum variance**

PCA finds orthogonal principal components (eigenvectors of the covariance matrix) ordered by variance explained, and projects data onto the top components to reduce dimensionality while retaining most variance. It is unsupervised and uses no labels.

</details>

**Q17.** What is the role of the learning rate in gradient descent?  ⭐ _(basic)_

- A. It measures the model's accuracy
- B. It controls the step size of each parameter update
- C. It sets the train/test split ratio
- D. It defines the number of features

<details>
<summary>Show answer</summary>

**B. It controls the step size of each parameter update**

The learning rate scales how far parameters move along the negative gradient each step. Too large can overshoot/diverge; too small makes convergence very slow.

</details>

**Q18.** Which statement correctly contrasts batch, stochastic (SGD), and mini-batch gradient descent?  ⭐ _(intermediate)_

- A. Batch uses the entire dataset per update; SGD uses one sample per update; mini-batch uses a small subset per update
- B. Batch uses one sample; SGD uses the whole dataset; mini-batch uses two samples
- C. All three update parameters using exactly one sample
- D. Mini-batch never converges, unlike the other two

<details>
<summary>Show answer</summary>

**A. Batch uses the entire dataset per update; SGD uses one sample per update; mini-batch uses a small subset per update**

Batch GD computes the gradient over all data (stable but slow/memory-heavy), SGD updates per single example (noisy, fast), and mini-batch uses small batches — the common compromise giving stable yet efficient updates.

</details>

**Q19.** During training you set the learning rate very high and the loss diverges (increases or oscillates wildly). What is the most appropriate fix? _(intermediate)_

- A. Increase the learning rate further
- B. Decrease the learning rate
- C. Remove all features
- D. Switch from classification to regression

<details>
<summary>Show answer</summary>

**B. Decrease the learning rate**

A too-large learning rate causes gradient descent to overshoot the minimum and diverge. Lowering it makes steps small enough to converge; learning-rate decay/scheduling also helps.

</details>

### 4.4 Python, NumPy, pandas & scikit-learn

> Python language MCQs, NumPy array ops/broadcasting/output prediction, pandas groupby/merge/indexing/output prediction, scikit-learn API (fit/transform/predict, Pipeline, train_test_split, random_state, fit only on train).
>
> **16 questions**, 14 ⭐ high-yield.

**Q1.** What is the output of the following code?

```python
import numpy as np
a = np.array([[1], [2], [3]])
b = np.array([1, 2, 3])
print(a + b)
```  ⭐ _(intermediate)_

- A. [[2 3 4]
 [3 4 5]
 [4 5 6]]
- B. [[2 4 6]]
- C. [2 4 6]
- D. ValueError: operands could not be broadcast together

<details>
<summary>Show answer</summary>

**A. [[2 3 4]
 [3 4 5]
 [4 5 6]]**

Broadcasting aligns the (3,1) column with the (3,) row, stretching both to (3,3): element [i,j] = a[i] + b[j]. Shapes are compatible because each mismatched dimension is 1.

</details>

**Q2.** In NumPy, which statement about basic slicing (e.g. `arr[1:4]`) versus advanced/fancy indexing (e.g. `arr[[1,2,3]]`) is correct?  ⭐ _(intermediate)_

- A. Basic slicing returns a view (shares memory); fancy indexing returns a copy
- B. Both always return copies
- C. Both always return views
- D. Basic slicing returns a copy; fancy indexing returns a view

<details>
<summary>Show answer</summary>

**A. Basic slicing returns a view (shares memory); fancy indexing returns a copy**

A basic slice is a view, so modifying it mutates the original array. Advanced (integer/boolean) indexing returns a new copy, so writes to it do not affect the source.

</details>

**Q3.** A store's daily unit-sales sit in `arr = np.arange(12).reshape(3, 4)`. What does `arr[:2, 1:3]` return?  ⭐ _(intermediate)_

- A. [[1 2]
 [5 6]]
- B. [[0 1]
 [4 5]]
- C. [[1 2 3]
 [5 6 7]]
- D. [[4 5]
 [8 9]]

<details>
<summary>Show answer</summary>

**A. [[1 2]
 [5 6]]**

Rows 0-1 and columns 1-2 (stop indices excluded). Row 0 is [0 1 2 3] -> [1 2]; row 1 is [4 5 6 7] -> [5 6]. Verified by execution.

</details>

**Q4.** What is the resulting shape of `np.ones((3, 1)) + np.arange(3)`?  ⭐ _(intermediate)_

- A. (3, 3)
- B. (3, 1)
- C. (1, 3)
- D. ValueError

<details>
<summary>Show answer</summary>

**A. (3, 3)**

Shapes (3,1) and (3,) broadcast: the (3,) is treated as (1,3), and the size-1 dims stretch to give (3,3). Verified by execution.

</details>

**Q5.** Which NumPy expression computes the column-wise mean of a 2D array `X` of shape (n_samples, n_features), returning one value per feature?  ⭐ _(basic)_

- A. X.mean(axis=0)
- B. X.mean(axis=1)
- C. X.mean()
- D. np.mean(X, keepdims=True)

<details>
<summary>Show answer</summary>

**A. X.mean(axis=0)**

axis=0 collapses the rows (samples), leaving one mean per column (feature), shape (n_features,). axis=1 would average across features for each sample instead.

</details>

**Q6.** What does `np.arange(6).reshape(2, 3).T.shape` evaluate to? _(basic)_

- A. (3, 2)
- B. (2, 3)
- C. (6,)
- D. (3, 3)

<details>
<summary>Show answer</summary>

**A. (3, 2)**

reshape(2,3) gives a 2x3 array; `.T` transposes it to 3x2. Verified by execution.

</details>

**Q7.** Given a SKU price array `p = np.array([10, 20, 30, 40])`, what is the output of `print(p[p > 15].sum())`? _(basic)_

- A. 90
- B. 100
- C. 60
- D. [20 30 40]

<details>
<summary>Show answer</summary>

**A. 90**

Boolean mask `p > 15` selects [20, 30, 40]; their sum is 90. Verified by execution.

</details>

**Q8.** For a DataFrame `df` of apparel sales, which call returns total quantity per `category` as a Series indexed by category?  ⭐ _(basic)_

- A. df.groupby('category')['quantity'].sum()
- B. df['quantity'].sum('category')
- C. df.groupby('quantity')['category'].sum()
- D. df.agg('category')['quantity'].sum()

<details>
<summary>Show answer</summary>

**A. df.groupby('category')['quantity'].sum()**

groupby('category') forms one group per category; selecting the 'quantity' column and summing yields a Series indexed by category.

</details>

**Q9.** What is the key difference between `df.loc[]` and `df.iloc[]`?  ⭐ _(basic)_

- A. .loc is label-based; .iloc is integer-position-based
- B. .loc is integer-position-based; .iloc is label-based
- C. .loc selects rows only; .iloc selects columns only
- D. They are aliases and behave identically

<details>
<summary>Show answer</summary>

**A. .loc is label-based; .iloc is integer-position-based**

.loc selects by index/column labels (and is inclusive of the stop label in slices); .iloc selects by integer position (stop exclusive, like Python slicing).

</details>

**Q10.** `left` has `id` values [1, 2] and `right` has `id` values [2, 3]. After `pd.merge(left, right, on='id')` (default join), how many rows does the result have?  ⭐ _(intermediate)_

- A. 1
- B. 2
- C. 3
- D. 4

<details>
<summary>Show answer</summary>

**A. 1**

pd.merge defaults to an inner join, keeping only keys present in both frames. Only id=2 matches, so the result has 1 row. Verified by execution.

</details>

**Q11.** Which pandas snippet correctly filters `df` to rows where `price` exceeds 100 AND `in_stock` is True?  ⭐ _(intermediate)_

- A. df[(df['price'] > 100) & (df['in_stock'])]
- B. df[df['price'] > 100 and df['in_stock']]
- C. df[df['price'] > 100 & df['in_stock']]
- D. df.filter(price > 100 and in_stock)

<details>
<summary>Show answer</summary>

**A. df[(df['price'] > 100) & (df['in_stock'])]**

Pandas boolean masks must use element-wise `&` with each condition parenthesized because `&` binds tighter than the comparison operators. The Python `and` keyword raises a ValueError on a Series.

</details>

**Q12.** In scikit-learn, when scaling features with `StandardScaler`, what is the correct way to avoid data leakage?  ⭐ _(intermediate)_

- A. Call fit (or fit_transform) on the training set only, then transform the test set with that fitted scaler
- B. Call fit_transform on the full dataset before splitting
- C. Call fit_transform separately on train and on test
- D. Call fit on the test set and transform the train set

<details>
<summary>Show answer</summary>

**A. Call fit (or fit_transform) on the training set only, then transform the test set with that fitted scaler**

Statistics (mean/std) must be learned from training data alone; the same fitted scaler then transforms the test set. Fitting on full data or on the test set leaks test information into preprocessing.

</details>

**Q13.** What is the primary purpose of a scikit-learn `Pipeline`?  ⭐ _(intermediate)_

- A. To chain preprocessing steps and a final estimator so that fit/predict apply each step in order and prevent leakage during cross-validation
- B. To run multiple models in parallel on different CPU cores
- C. To automatically tune hyperparameters without a search
- D. To convert a pandas DataFrame into a NumPy array

<details>
<summary>Show answer</summary>

**A. To chain preprocessing steps and a final estimator so that fit/predict apply each step in order and prevent leakage during cross-validation**

A Pipeline bundles transformers + an estimator into one object; during cross-validation each fold re-fits the transformers on its own training portion, preventing leakage. It does not itself do parallelism or hyperparameter search.

</details>

**Q14.** What does the `random_state` parameter in `train_test_split` control?  ⭐ _(basic)_

- A. It seeds the shuffling so the split is reproducible across runs
- B. It sets the proportion of data assigned to the test set
- C. It enables stratified sampling by the target
- D. It chooses the model's initialization weights

<details>
<summary>Show answer</summary>

**A. It seeds the shuffling so the split is reproducible across runs**

random_state fixes the random seed used to shuffle before splitting, giving the same split every run. test_size sets the proportion; stratify enables stratification; neither relates to model weights.

</details>

**Q15.** In the scikit-learn estimator API, which method does an unsupervised transformer like `PCA` or `StandardScaler` use to learn parameters AND apply them in one call?  ⭐ _(basic)_

<details>
<summary>Show answer</summary>

**fit_transform**

fit_transform learns the parameters (fit) and returns the transformed data (transform) in a single call; it is typically used on the training set only.

</details>

**Q16.** Select ALL true statements about the scikit-learn estimator API.  ⭐ _(intermediate)_

- A. Supervised estimators implement fit(X, y) and predict(X)
- B. transform(X) is implemented by transformers (e.g. scalers, encoders) to produce transformed features
- C. You should call fit on the test set to evaluate generalization
- D. predict_proba returns class probabilities for classifiers that support it

<details>
<summary>Show answer</summary>

**A, B, D — Supervised estimators implement fit(X, y) and predict(X); transform(X) is implemented by transformers to produce transformed features; predict_proba returns class probabilities for classifiers that support it**

A, B, and D describe the standard API correctly. C is wrong: models are fit only on training data and evaluated on the test set with predict — fitting on test data leaks information and invalidates the evaluation.

</details>

### 4.5 Statistics & Probability

> Mean/median/mode/variance/std, normal & other distributions, conditional probability, Bayes theorem, hypothesis testing, p-value, type I/II error, central limit theorem, correlation vs causation, covariance.
>
> **16 questions**, 13 ⭐ high-yield.

**Q1.** A retail analyst records daily units sold for a jacket over 7 days: 12, 15, 15, 18, 20, 22, 90. The last value (90) is a clearance-day spike. Which measure of central tendency best represents a 'typical' day?  ⭐ _(basic)_

- A. Mean, because it uses every data point
- B. Median, because it is robust to the outlier
- C. Mode, because 15 occurs most often
- D. Range, because it captures the spread

<details>
<summary>Show answer</summary>

**B. Median, because it is robust to the outlier**

Sorted: 12,15,15,18,20,22,90. Median = 4th value = 18; mean ≈ 27.4 is inflated by the 90 outlier. The median is robust to extreme values and better reflects a typical day. Range is a spread measure, not central tendency.

</details>

**Q2.** For the data set {4, 8, 8, 12} the population variance is computed as the average of squared deviations from the mean. What is the population variance? _(intermediate)_

- A. 8
- B. 16
- C. 4
- D. 5.33

<details>
<summary>Show answer</summary>

**A. 8**

Mean = 8. Squared deviations: 16, 0, 0, 16 = 32; population variance = 32/4 = 8. Dividing by (n-1)=3 would give the sample variance (~10.67), the common distractor.

</details>

**Q3.** In a normal distribution, approximately what percentage of values fall within one standard deviation of the mean?  ⭐ _(basic)_

- A. 50%
- B. 68%
- C. 95%
- D. 99.7%

<details>
<summary>Show answer</summary>

**B. 68%**

By the empirical (68-95-99.7) rule, ~68% lie within ±1 SD, ~95% within ±2 SD, and ~99.7% within ±3 SD of the mean.

</details>

**Q4.** A store's daily footfall is normally distributed with mean 500 and standard deviation 50. Roughly what fraction of days see footfall above 600?  ⭐ _(intermediate)_

- A. About 2.5%
- B. About 16%
- C. About 5%
- D. About 32%

<details>
<summary>Show answer</summary>

**A. About 2.5%**

600 is exactly 2 SD above the mean (z = (600-500)/50 = 2). About 95% lie within ±2 SD, so ~2.5% lie in each tail beyond ±2 SD.

</details>

**Q5.** Bayes' theorem is correctly stated as:  ⭐ _(basic)_

- A. P(A|B) = P(A) · P(B)
- B. P(A|B) = P(B|A) · P(A) / P(B)
- C. P(A|B) = P(A|B) · P(B) / P(A)
- D. P(A|B) = P(A) + P(B) − P(A∩B)

<details>
<summary>Show answer</summary>

**B. P(A|B) = P(B|A) · P(A) / P(B)**

Bayes' theorem updates the prior P(A) using the likelihood P(B|A) and evidence P(B). Option A is the independence rule and option D is the addition rule for unions, not Bayes.

</details>

**Q6.** A fraud filter for online apparel orders flags 1% of orders as fraud (prior). The detector catches 90% of true fraud (sensitivity) and falsely flags 5% of legitimate orders. Given an order is flagged, what is the probability it is actually fraud?  ⭐ _(advanced)_

- A. About 90%
- B. About 15%
- C. About 50%
- D. About 5%

<details>
<summary>Show answer</summary>

**B. About 15%**

P(fraud|flag) = (0.90·0.01) / (0.90·0.01 + 0.05·0.99) = 0.009 / 0.0585 ≈ 0.154. The low base rate (1%) makes most flags false positives despite high sensitivity — the classic base-rate fallacy.

</details>

**Q7.** Two cards are drawn without replacement from a standard 52-card deck. What is the probability both are kings? _(intermediate)_

- A. (4/52) · (4/52)
- B. (4/52) · (3/51)
- C. (4/52) + (3/51)
- D. (1/13) · (1/13)

<details>
<summary>Show answer</summary>

**B. (4/52) · (3/51)**

Without replacement, after drawing one king there are 3 kings left in 51 cards, so P = (4/52)·(3/51) = 1/221. Multiplying (4/52)·(4/52) would incorrectly assume replacement.

</details>

**Q8.** The Central Limit Theorem states that, for a sufficiently large sample size, the sampling distribution of the sample mean:  ⭐ _(intermediate)_

- A. Becomes uniform regardless of the population
- B. Approaches a normal distribution regardless of the population's shape
- C. Has the same standard deviation as the population
- D. Equals the population distribution exactly

<details>
<summary>Show answer</summary>

**B. Approaches a normal distribution regardless of the population's shape**

The CLT says the distribution of sample means tends toward normal as n grows, even when the population is skewed (assuming finite variance). The standard error (σ/√n) is smaller than the population SD, so option C is wrong.

</details>

**Q9.** In hypothesis testing, what does the p-value represent?  ⭐ _(intermediate)_

- A. The probability that the null hypothesis is true
- B. The probability of observing data at least as extreme as the sample, assuming the null hypothesis is true
- C. The probability that the alternative hypothesis is true
- D. The size of the effect being measured

<details>
<summary>Show answer</summary>

**B. The probability of observing data at least as extreme as the sample, assuming the null hypothesis is true**

The p-value is computed under H0 and measures how extreme the observed data are. It is NOT the probability that H0 is true (a common misinterpretation), nor a measure of effect size.

</details>

**Q10.** Using a significance level α = 0.05, you reject the null hypothesis when:  ⭐ _(basic)_

- A. p-value > 0.05
- B. p-value ≤ 0.05
- C. p-value = 1
- D. the sample size is large

<details>
<summary>Show answer</summary>

**B. p-value ≤ 0.05**

You reject H0 when the p-value is at most the significance level α, indicating the result is statistically significant. A p-value above α means you fail to reject H0.

</details>

**Q11.** A Type I error in hypothesis testing is best described as:  ⭐ _(intermediate)_

- A. Failing to reject a false null hypothesis
- B. Rejecting a true null hypothesis (false positive)
- C. Accepting a true alternative hypothesis
- D. Increasing the sample size unnecessarily

<details>
<summary>Show answer</summary>

**B. Rejecting a true null hypothesis (false positive)**

A Type I error (probability α) is rejecting H0 when it is actually true — a false positive. Failing to reject a false H0 is the Type II error (false negative).

</details>

**Q12.** A retailer A/B tests a new product page. Suppose the new page truly has NO effect on conversion, but the test concludes it improves conversion and the team rolls it out. Which error has occurred?  ⭐ _(intermediate)_

- A. Type I error
- B. Type II error
- C. Sampling error
- D. Measurement error

<details>
<summary>Show answer</summary>

**A. Type I error**

The null hypothesis (no effect) is actually true, yet it was rejected — a false positive, i.e., a Type I error. A Type II error would be missing a real improvement.

</details>

**Q13.** Which statement about correlation and causation is correct?  ⭐ _(intermediate)_

- A. A strong correlation always implies one variable causes the other
- B. Correlation can arise from a confounding variable without any causal link
- C. Causation cannot exist without correlation in raw data
- D. Zero correlation proves the variables are independent

<details>
<summary>Show answer</summary>

**B. Correlation can arise from a confounding variable without any causal link**

Correlation does not imply causation; a lurking confounder (e.g., a season driving both ice-cream and swimwear sales) can create correlation without a direct cause. Also, zero linear correlation does not prove independence (relationships can be nonlinear).

</details>

**Q14.** How does the Pearson correlation coefficient relate to covariance?  ⭐ _(intermediate)_

- A. They are identical in value
- B. Correlation is covariance divided by the product of the two standard deviations
- C. Covariance is always between −1 and +1; correlation is unbounded
- D. Correlation is covariance multiplied by the sample size

<details>
<summary>Show answer</summary>

**B. Correlation is covariance divided by the product of the two standard deviations**

Correlation = Cov(X,Y) / (σx·σy), the standardized (scale-free) version of covariance, bounded in [−1, 1]. Covariance itself is unbounded and depends on the variables' units.

</details>

**Q15.** A fair six-sided die is rolled once. What is the expected value of the outcome? _(basic)_

- A. 3
- B. 3.5
- C. 4
- D. 6

<details>
<summary>Show answer</summary>

**B. 3.5**

Expected value = (1+2+3+4+5+6)/6 = 21/6 = 3.5. The expected value need not be a value the die can actually show.

</details>

**Q16.** Select ALL statements that are TRUE about a distribution that is right-skewed (positively skewed), such as customer spend per order.  ⭐ _(advanced)_

- A. The mean is typically greater than the median
- B. The long tail extends to the right (higher values)
- C. The mean is typically less than the median
- D. Median is generally more robust to the skew than the mean

<details>
<summary>Show answer</summary>

**A, B, D. The mean is typically greater than the median; The long tail extends to the right (higher values); Median is generally more robust to the skew than the mean**

In a right-skewed distribution the long tail of high values pulls the mean above the median, so mean > median (making option C false). The median resists the influence of those extreme high values.

</details>

### 4.6 Deep Learning & Neural Networks

> Perceptron/MLP, activation functions and their output ranges (sigmoid/tanh/ReLU/softmax), loss functions, backpropagation, vanishing/exploding gradients, dropout, batch normalization, CNN basics (conv/pooling), RNN/LSTM, optimizers (SGD/Adam), epochs vs batch size.
>
> **16 questions**, 15 ⭐ high-yield.

**Q1.** Which activation function is most commonly used in the OUTPUT layer of a neural network that must assign each product image to exactly ONE of 12 apparel categories (e.g. shirts, dresses, shoes)?  ⭐ _(basic)_

- A. Sigmoid
- B. Softmax
- C. ReLU
- D. Tanh

<details>
<summary>Show answer</summary>

**B. Softmax**

Softmax produces a probability distribution over mutually exclusive classes that sums to 1, which is exactly what single-label multi-class classification needs. Sigmoid is for binary or multi-label outputs, and ReLU/Tanh are hidden-layer activations.

</details>

**Q2.** What is the output range of the sigmoid (logistic) activation function?  ⭐ _(basic)_

- A. (-1, 1)
- B. (0, 1)
- C. (0, ∞)
- D. (-∞, ∞)

<details>
<summary>Show answer</summary>

**B. (0, 1)**

Sigmoid squashes any real input into the open interval (0, 1), making it suitable for probabilities. Tanh outputs (-1, 1) and ReLU outputs [0, ∞).

</details>

**Q3.** What is the output range of the tanh activation function?  ⭐ _(basic)_

- A. (0, 1)
- B. (-1, 1)
- C. (0, ∞)
- D. [0, 1]

<details>
<summary>Show answer</summary>

**B. (-1, 1)**

Tanh is a zero-centered sigmoid-like function bounded between -1 and 1, which often helps optimization converge faster than the (0,1) sigmoid. The (0,1) range belongs to sigmoid.

</details>

**Q4.** Which activation function outputs 0 for all negative inputs and the input itself for positive inputs?  ⭐ _(basic)_

- A. Sigmoid
- B. Tanh
- C. ReLU
- D. Softmax

<details>
<summary>Show answer</summary>

**C. ReLU**

ReLU is defined as f(x) = max(0, x): it zeroes out negatives and passes positives unchanged. This sparsity and non-saturating positive region make it the default hidden-layer activation.

</details>

**Q5.** A single-layer perceptron (with a step/linear activation) CANNOT learn which of the following functions?  ⭐ _(intermediate)_

- A. AND
- B. OR
- C. NOT
- D. XOR

<details>
<summary>Show answer</summary>

**D. XOR**

XOR is not linearly separable, so a single-layer perceptron cannot represent it; a multi-layer perceptron with a hidden layer is required. AND, OR, and NOT are all linearly separable.

</details>

**Q6.** The vanishing gradient problem in deep networks is MOST strongly associated with which activation functions?  ⭐ _(intermediate)_

- A. ReLU and Leaky ReLU
- B. Sigmoid and Tanh
- C. Softmax and ReLU
- D. Linear and ReLU

<details>
<summary>Show answer</summary>

**B. Sigmoid and Tanh**

Sigmoid and tanh saturate at their extremes where derivatives approach 0, so gradients shrink toward zero as they propagate back through many layers. ReLU mitigates this because its derivative is 1 for positive inputs.

</details>

**Q7.** Which loss function is the standard choice for training a multi-class single-label classifier (e.g. tagging an item with one apparel category)?  ⭐ _(intermediate)_

- A. Mean Squared Error
- B. Categorical cross-entropy
- C. Hinge loss
- D. Mean Absolute Error

<details>
<summary>Show answer</summary>

**B. Categorical cross-entropy**

Categorical cross-entropy pairs naturally with a softmax output to penalize the predicted probability of the true class, and it gives stronger gradients than MSE for classification. MSE/MAE are regression losses.

</details>

**Q8.** What is the primary purpose of the backpropagation algorithm?  ⭐ _(intermediate)_

- A. To initialize the network weights randomly
- B. To compute the gradients of the loss with respect to each weight so they can be updated
- C. To select the number of layers in the network
- D. To normalize the inputs before training

<details>
<summary>Show answer</summary>

**B. To compute the gradients of the loss with respect to each weight so they can be updated**

Backpropagation applies the chain rule to efficiently compute the gradient of the loss with respect to every weight, which an optimizer then uses to update them. It does not initialize weights or choose architecture.

</details>

**Q9.** During training, dropout primarily helps a neural network by:  ⭐ _(intermediate)_

- A. Reducing overfitting by randomly deactivating neurons each step
- B. Speeding up matrix multiplication
- C. Guaranteeing the gradient never vanishes
- D. Reducing the number of parameters permanently

<details>
<summary>Show answer</summary>

**A. Reducing overfitting by randomly deactivating neurons each step**

Dropout randomly zeroes a fraction of neurons during each training step, preventing co-adaptation and acting as regularization that improves generalization. It does not permanently remove parameters; all neurons are active at inference (with scaling).

</details>

**Q10.** What is the main benefit of batch normalization?  ⭐ _(intermediate)_

- A. It permanently removes neurons to reduce model size
- B. It normalizes layer activations, stabilizing and accelerating training
- C. It replaces the need for an activation function
- D. It guarantees zero training error

<details>
<summary>Show answer</summary>

**B. It normalizes layer activations, stabilizing and accelerating training**

Batch normalization standardizes the activations within a mini-batch, which allows higher learning rates, speeds convergence, and adds mild regularization. It does not remove neurons or replace activations.

</details>

**Q11.** In a convolutional neural network, what is the main purpose of a pooling layer (e.g. max pooling)?  ⭐ _(basic)_

- A. To add non-linearity to the network
- B. To downsample feature maps, reducing spatial dimensions and computation
- C. To normalize the pixel intensities of the input image
- D. To fully connect all neurons across layers

<details>
<summary>Show answer</summary>

**B. To downsample feature maps, reducing spatial dimensions and computation**

Pooling reduces the height and width of feature maps, lowering computation and providing some translation invariance. Non-linearity comes from activation functions, not pooling.

</details>

**Q12.** Which architecture is specifically designed to mitigate the vanishing gradient problem when modeling long-range dependencies in sequences?  ⭐ _(intermediate)_

- A. Vanilla (simple) RNN
- B. LSTM
- C. Standard feed-forward MLP
- D. Convolutional layer

<details>
<summary>Show answer</summary>

**B. LSTM**

LSTMs use a gated cell state (input, forget, and output gates) that lets gradients flow over many time steps, addressing the vanishing-gradient weakness of vanilla RNNs on long sequences.

</details>

**Q13.** Which optimizer combines momentum with per-parameter adaptive learning rates using estimates of the first and second moments of the gradients?  ⭐ _(intermediate)_

- A. Vanilla SGD
- B. Adam
- C. Plain Gradient Descent
- D. Newton's method

<details>
<summary>Show answer</summary>

**B. Adam**

Adam (Adaptive Moment Estimation) maintains running averages of the gradient (first moment) and its square (second moment) to adapt the step size per parameter, typically converging faster than plain SGD. Vanilla SGD uses a single fixed learning rate without adaptive moments.

</details>

**Q14.** In neural network training, what does one 'epoch' mean?  ⭐ _(basic)_

- A. One forward and backward pass over a single training example
- B. One update of the weights using a single mini-batch
- C. One complete pass through the entire training dataset
- D. The total number of layers in the network

<details>
<summary>Show answer</summary>

**C. One complete pass through the entire training dataset**

An epoch is one full sweep over all training samples. A single weight update on a mini-batch is one iteration/step, and many such steps make up one epoch.

</details>

**Q15.** If a training set has 10,000 product images and the batch size is 250, how many weight-update iterations occur in ONE epoch? _(intermediate)_

- A. 25
- B. 40
- C. 250
- D. 10,000

<details>
<summary>Show answer</summary>

**B. 40**

Iterations per epoch = total samples / batch size = 10,000 / 250 = 40. Each iteration processes one mini-batch and performs one weight update.

</details>

**Q16.** Select ALL techniques that are commonly used to combat the vanishing gradient problem in deep networks.  ⭐ _(advanced)_

- A. Using ReLU instead of sigmoid in hidden layers
- B. Adding more sigmoid layers to deepen the network
- C. Using residual/skip connections
- D. Applying batch normalization

<details>
<summary>Show answer</summary>

**A, C, D. Using ReLU instead of sigmoid in hidden layers; Using residual/skip connections; Applying batch normalization**

ReLU keeps a derivative of 1 for positive inputs, residual connections give gradients a shortcut path, and batch normalization keeps activations in a healthy range; all help gradients flow. Stacking more sigmoid layers worsens the problem.

</details>

### 4.7 NLP, LLMs & Generative AI (2026)

> Tokenization, bag-of-words/TF-IDF, word2vec/embeddings, transformers & self-attention, BERT vs GPT (encoder vs decoder), fine-tuning vs RAG vs prompting, hallucination, temperature/top-p, vector databases & semantic search, what an LLM/agent is.
>
> **16 questions**, 14 ⭐ high-yield.

**Q1.** In a bag-of-words (BoW) representation of text, which key piece of information is discarded?  ⭐ _(basic)_

- A. The frequency count of each word
- B. The order/position of words in the document
- C. The vocabulary of the corpus
- D. The presence or absence of a word

<details>
<summary>Show answer</summary>

**B. The order/position of words in the document**

BoW represents a document as an unordered multiset of token counts, so word order and syntax are lost ('great not bad' and 'bad not great' map to the same vector). Frequency counts and vocabulary are exactly what BoW keeps.

</details>

**Q2.** In TF-IDF, the inverse-document-frequency (IDF) term assigns the LOWEST weight to a word that:  ⭐ _(intermediate)_

- A. Appears in only one document of the corpus
- B. Appears in almost every document of the corpus
- C. Is very long in character length
- D. Never appears in the corpus at all

<details>
<summary>Show answer</summary>

**B. Appears in almost every document of the corpus**

IDF = log(N / df). A word appearing in nearly all documents has a large df, driving IDF toward zero, so common words (e.g. 'shirt' in an apparel catalogue) are down-weighted while rare, discriminative words get high weight. (A word that never appears has df=0 and no defined TF-IDF contribution, so it is not the intended 'lowest weight' case.)

</details>

**Q3.** What is the main advantage of Word2Vec embeddings over a TF-IDF / bag-of-words representation?  ⭐ _(basic)_

- A. They produce sparse, high-dimensional one-hot vectors
- B. They capture semantic similarity, so related words have nearby vectors
- C. They require no training data at all
- D. They guarantee the exact word counts are preserved

<details>
<summary>Show answer</summary>

**B. They capture semantic similarity, so related words have nearby vectors**

Word2Vec learns dense, low-dimensional vectors where semantically related words ('king'/'queen', 'jacket'/'coat') sit close together, unlike sparse TF-IDF vectors that treat every word as orthogonal and carry no notion of meaning.

</details>

**Q4.** The famous Word2Vec analogy vector('king') - vector('man') + vector('woman') results in a vector closest to which word?  ⭐ _(basic)_

- A. prince
- B. queen
- C. royal
- D. throne

<details>
<summary>Show answer</summary>

**B. queen**

Word2Vec embeddings encode relationships as consistent vector offsets; the gender direction added to 'king' lands nearest 'queen'. This is the canonical demonstration that the embedding space captures linear semantic structure.

</details>

**Q5.** What is the core operation performed by the self-attention mechanism in a Transformer?  ⭐ _(intermediate)_

- A. It convolves a fixed-size filter over the token sequence
- B. It lets each token compute a weighted sum of all tokens' value vectors using query-key similarity
- C. It recurrently passes a hidden state from one token to the next
- D. It clusters tokens by their TF-IDF scores before pooling

<details>
<summary>Show answer</summary>

**B. It lets each token compute a weighted sum of all tokens' value vectors using query-key similarity**

Self-attention scores every token pair via Query-Key dot products (scaled and softmaxed), then each token's output is the weighted sum of all Value vectors. This is what lets Transformers model long-range dependencies without recurrence or convolution.

</details>

**Q6.** In scaled dot-product attention, the query-key dot products are divided by the square root of the key dimension (sqrt(d_k)). Why? _(advanced)_

- A. To make the attention matrix symmetric
- B. To keep the dot-product magnitudes from growing large and pushing softmax into tiny-gradient regions
- C. To reduce the number of trainable parameters
- D. To convert the scores into integer token IDs

<details>
<summary>Show answer</summary>

**B. To keep the dot-product magnitudes from growing large and pushing softmax into tiny-gradient regions**

For inputs with unit-variance independent components, a d_k-dimensional dot product has variance d_k; dividing by sqrt(d_k) rescales the variance back to ~1. Without it, large dot products push softmax toward a near one-hot distribution with vanishingly small gradients, destabilizing training.

</details>

**Q7.** BERT and GPT differ primarily in their Transformer architecture. Which statement is correct?  ⭐ _(intermediate)_

- A. BERT is an encoder (bidirectional) model; GPT is a decoder (autoregressive, left-to-right) model
- B. BERT is a decoder model; GPT is an encoder model
- C. Both are encoder-only models trained identically
- D. Neither uses the Transformer architecture

<details>
<summary>Show answer</summary>

**A. BERT is an encoder (bidirectional) model; GPT is a decoder (autoregressive, left-to-right) model**

BERT uses a bidirectional Transformer encoder trained with masked-language modelling, making it strong for understanding tasks (classification, NER). GPT is a decoder-only model that predicts the next token left-to-right, making it strong for generation.

</details>

**Q8.** A retail team wants a model to CLASSIFY incoming customer reviews by sentiment, using rich bidirectional context. Which model family is the most natural fit? _(intermediate)_

- A. A GPT-style decoder-only model for next-token generation
- B. A BERT-style encoder model fine-tuned for classification
- C. A bag-of-words model with no embeddings
- D. An image diffusion model

<details>
<summary>Show answer</summary>

**B. A BERT-style encoder model fine-tuned for classification**

Encoder models like BERT read the whole sentence bidirectionally and produce a pooled representation ideal for classification tasks such as sentiment. Decoder-only GPT models are optimized for generation, not the most natural pick for a pure classifier.

</details>

**Q9.** How does increasing the TEMPERATURE parameter affect an LLM's text generation?  ⭐ _(basic)_

- A. It makes the output more deterministic and repetitive
- B. It flattens the probability distribution, making output more random/diverse
- C. It permanently changes the model's weights
- D. It reduces the size of the model's vocabulary

<details>
<summary>Show answer</summary>

**B. It flattens the probability distribution, making output more random/diverse**

Temperature scales the logits before softmax: low values (near 0) sharpen the distribution toward greedy, deterministic output; higher values flatten it, increasing randomness and diversity. Use low temperature for factual tasks, higher for creative copy.

</details>

**Q10.** What does top-p (nucleus) sampling do during generation?  ⭐ _(intermediate)_

- A. Always selects the single highest-probability token
- B. Samples from the smallest set of tokens whose cumulative probability exceeds p
- C. Selects exactly p tokens regardless of their probabilities
- D. Removes the p most likely tokens before sampling

<details>
<summary>Show answer</summary>

**B. Samples from the smallest set of tokens whose cumulative probability exceeds p**

Top-p (nucleus) sampling keeps the smallest candidate set whose cumulative probability mass reaches p and samples from it, so the candidate pool adapts to the distribution's shape. Top-k, by contrast, fixes the count of candidates at k.

</details>

**Q11.** An LLM-based assistant for an apparel brand must answer questions about TODAY's inventory and prices, which change daily. Which approach is most appropriate?  ⭐ _(intermediate)_

- A. Fine-tune the base model nightly on the full catalogue
- B. Use Retrieval-Augmented Generation (RAG) to fetch current data at query time
- C. Raise the temperature so the model guesses prices
- D. Increase the context window and hope the model memorizes prices

<details>
<summary>Show answer</summary>

**B. Use Retrieval-Augmented Generation (RAG) to fetch current data at query time**

RAG retrieves up-to-date external data at inference and grounds the answer, which is ideal for frequently changing, auditable knowledge like live inventory. Fine-tuning bakes knowledge into weights and is costly to repeat for daily updates.

</details>

**Q12.** Which of the following are valid reasons to choose RAG over fine-tuning? (Select all that apply.)  ⭐ _(intermediate)_

- A. The knowledge base changes frequently and must stay current
- B. You need source attribution / auditability of the answer
- C. You want to permanently change the model's writing style and tone with thousands of labeled examples
- D. You want to add proprietary documents without retraining the model

<details>
<summary>Show answer</summary>

**A, B, D. (The knowledge base changes frequently and must stay current; You need source attribution / auditability of the answer; You want to add proprietary documents without retraining the model)**

RAG shines when knowledge is dynamic, must be auditable/cited, and can be added without retraining. Permanently changing style/tone/format with many labeled examples is the classic fine-tuning use case, not RAG.

</details>

**Q13.** In the context of LLMs, what does 'hallucination' refer to?  ⭐ _(basic)_

- A. The model refusing to answer a question
- B. The model generating fluent but factually incorrect or fabricated content
- C. The model running out of context window
- D. The model returning the input unchanged

<details>
<summary>Show answer</summary>

**B. The model generating fluent but factually incorrect or fabricated content**

Hallucination is when an LLM produces confident, plausible-sounding output that is not grounded in fact or in the provided sources. Grounding with RAG, lower temperature, and faithfulness checks help reduce it.

</details>

**Q14.** In a semantic-search / RAG pipeline, what does a vector database primarily store and search over?  ⭐ _(basic)_

- A. Raw SQL rows matched by exact keyword equality
- B. Dense embedding vectors, searched by nearest-neighbor (similarity) lookup
- C. Compressed image files indexed by filename
- D. The model's weight matrices

<details>
<summary>Show answer</summary>

**B. Dense embedding vectors, searched by nearest-neighbor (similarity) lookup**

A vector database stores embedding vectors and retrieves the closest ones via approximate nearest-neighbor search (e.g. cosine/dot-product similarity), enabling semantic retrieval where meaning, not exact keywords, drives the match.

</details>

**Q15.** Which similarity metric is most commonly used to compare two text embedding vectors for semantic search?  ⭐ _(intermediate)_

- A. Cosine similarity
- B. Hamming distance
- C. Jaccard index on raw characters
- D. Edit (Levenshtein) distance

<details>
<summary>Show answer</summary>

**A. Cosine similarity**

Cosine similarity measures the angle between embedding vectors, capturing directional (semantic) closeness while being insensitive to vector magnitude, which is why it is the default for comparing dense text embeddings. Hamming/edit distances operate on discrete symbols, not dense vectors.

</details>

**Q16.** Which description best defines an LLM-based 'agent' (as opposed to a single LLM call)?  ⭐ _(intermediate)_

- A. An LLM that only completes one prompt and stops
- B. A system where an LLM plans steps and invokes external tools/actions in a loop to achieve a goal
- C. A fixed rule-based chatbot with no language model
- D. A vector database used for storage only

<details>
<summary>Show answer</summary>

**B. A system where an LLM plans steps and invokes external tools/actions in a loop to achieve a goal**

An agent uses an LLM as a reasoning engine that decides actions, calls tools (search, APIs, code), observes results, and iterates toward a goal, rather than producing a single one-shot completion. This planning-plus-tool-use loop is the defining characteristic.

</details>

### 4.8 SQL & Data Handling

> SELECT/WHERE/GROUP BY/HAVING/ORDER BY, INNER vs LEFT JOIN, aggregate functions, window functions (ROW_NUMBER/RANK), subqueries, predicting query output on a small retail schema (orders, customers, products), basic data cleaning.
>
> **16 questions**, 14 ⭐ high-yield.

**Q1.** In a SQL query, which clause is used to filter rows BEFORE any grouping or aggregation is performed?  ⭐ _(basic)_

- A. WHERE
- B. HAVING
- C. GROUP BY
- D. ORDER BY

<details>
<summary>Show answer</summary>

**A. WHERE**

WHERE filters individual rows before grouping/aggregation; HAVING filters groups after aggregation. HAVING cannot be used to filter pre-aggregation rows and WHERE cannot reference aggregate functions like SUM().

</details>

**Q2.** Given an orders table, which query correctly returns only those customer_id values whose total order count is greater than 5?  ⭐ _(basic)_

- A. SELECT customer_id FROM orders WHERE COUNT(*) > 5 GROUP BY customer_id
- B. SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 5
- C. SELECT customer_id, COUNT(*) FROM orders WHERE COUNT(*) > 5
- D. SELECT customer_id FROM orders GROUP BY customer_id WHERE COUNT(*) > 5

<details>
<summary>Show answer</summary>

**B. SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 5**

Filtering on an aggregate (COUNT(*)) requires HAVING after GROUP BY. WHERE cannot contain aggregate functions, and a WHERE clause cannot follow GROUP BY, which eliminates the other options.

</details>

**Q3.** A customers table has 100 rows and an orders table has 250 rows. Every order references a valid customer, but only 80 customers have placed at least one order. How many rows does 'SELECT * FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id' return?  ⭐ _(intermediate)_

- A. 250
- B. 270
- C. 100
- D. 350

<details>
<summary>Show answer</summary>

**B. 270**

The 250 order rows all match a customer, so they are returned, plus the 20 customers with no orders each produce one row (with NULL order columns), giving 250 + 20 = 270. A LEFT JOIN keeps all left-table rows.

</details>

**Q4.** What is the key difference between an INNER JOIN and a LEFT JOIN?  ⭐ _(basic)_

- A. INNER JOIN returns all rows from the left table; LEFT JOIN returns only matching rows
- B. INNER JOIN returns only matching rows from both tables; LEFT JOIN also returns unmatched rows from the left table with NULLs
- C. They always return the same number of rows
- D. LEFT JOIN removes duplicate rows while INNER JOIN keeps them

<details>
<summary>Show answer</summary>

**B. INNER JOIN returns only matching rows from both tables; LEFT JOIN also returns unmatched rows from the left table with NULLs**

INNER JOIN keeps only rows with a match on both sides; LEFT JOIN keeps every left-table row, filling right-table columns with NULL when there is no match.

</details>

**Q5.** A products table contains a 'category' column. Some rows have category = NULL. Consider: SELECT category, COUNT(*) FROM products GROUP BY category. Which statement about NULL handling is correct?  ⭐ _(intermediate)_

- A. All NULL categories are excluded entirely from the result
- B. COUNT(*) returns 0 for the NULL group
- C. All rows with NULL category are grouped together into a single group
- D. The query raises an error because you cannot GROUP BY a nullable column

<details>
<summary>Show answer</summary>

**C. All rows with NULL category are grouped together into a single group**

GROUP BY treats all NULLs as equal for grouping purposes, so they form one group, and COUNT(*) counts every row in that group (NULLs included). Note COUNT(category) would instead ignore the NULLs.

</details>

**Q6.** Which aggregate function call ignores NULL values when computing its result over the 'discount' column?  ⭐ _(intermediate)_

- A. COUNT(*)
- B. AVG(discount)
- C. COUNT(1)
- D. All of these count NULLs

<details>
<summary>Show answer</summary>

**B. AVG(discount)**

AVG(discount) (like SUM, MIN, MAX, and COUNT(column)) ignores NULLs. COUNT(*) and COUNT(1) count every row regardless of NULLs.

</details>

**Q7.** Consider a sales table:

product | amount
--------|-------
Shirt   | 100
Shirt   | 200
Jeans   | 300
Jeans   | NULL

What does 'SELECT product, AVG(amount) FROM sales GROUP BY product' return for Jeans?  ⭐ _(intermediate)_

- A. 150
- B. 300
- C. NULL
- D. 0

<details>
<summary>Show answer</summary>

**B. 300**

AVG ignores NULLs, so for Jeans it averages only the single non-NULL value 300 -> 300/1 = 300. It does NOT divide by 2.

</details>

**Q8.** What does the ROW_NUMBER() window function guarantee that RANK() does not?  ⭐ _(intermediate)_

- A. ROW_NUMBER assigns unique consecutive numbers even to tied rows, while RANK assigns the same number to ties
- B. ROW_NUMBER skips numbers after ties, while RANK does not
- C. ROW_NUMBER requires a GROUP BY clause, while RANK does not
- D. They are identical in all cases

<details>
<summary>Show answer</summary>

**A. ROW_NUMBER assigns unique consecutive numbers even to tied rows, while RANK assigns the same number to ties**

ROW_NUMBER() always produces distinct 1,2,3... values; RANK() gives tied rows the same rank and then skips the next value(s) (e.g. 1,1,3). DENSE_RANK would give 1,1,2.

</details>

**Q9.** Given the values 100, 100, 90, 80 ordered descending, what sequence does RANK() OVER (ORDER BY amount DESC) produce?  ⭐ _(intermediate)_

- A. 1, 2, 3, 4
- B. 1, 1, 2, 3
- C. 1, 1, 3, 4
- D. 1, 2, 2, 3

<details>
<summary>Show answer</summary>

**C. 1, 1, 3, 4**

RANK gives both 100s rank 1, then skips rank 2 because two rows occupied the top, so 90 gets rank 3 and 80 gets rank 4. DENSE_RANK would instead give 1,1,2,3.

</details>

**Q10.** To return the single most recent order per customer from an orders table (columns: customer_id, order_date), which window-function approach is correct?  ⭐ _(advanced)_

- A. Use ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) and keep rows where it equals 1
- B. Use COUNT(*) OVER (PARTITION BY customer_id) and keep rows where it equals 1
- C. Apply GROUP BY customer_id and SELECT order_date directly without aggregation
- D. Use ORDER BY order_date DESC LIMIT 1 without any partitioning

<details>
<summary>Show answer</summary>

**A. Use ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) and keep rows where it equals 1**

Partitioning by customer and ordering by date descending numbers each customer's orders; the row numbered 1 is that customer's latest order. A bare GROUP BY cannot return the full row's order_date unambiguously, and LIMIT 1 returns only one row overall.

</details>

**Q11.** In SQL's logical order of evaluation, which of these clauses is evaluated LAST?  ⭐ _(intermediate)_

- A. GROUP BY
- B. WHERE
- C. HAVING
- D. ORDER BY

<details>
<summary>Show answer</summary>

**D. ORDER BY**

Logical processing order is roughly FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY. Of the four options ORDER BY runs last, which is why column aliases defined in SELECT can be used in ORDER BY but not in WHERE.

</details>

**Q12.** Which subquery correctly returns customers who have NEVER placed an order, given customers(customer_id) and orders(customer_id)?  ⭐ _(intermediate)_

- A. SELECT customer_id FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)
- B. SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)
- C. SELECT customer_id FROM customers WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.customer_id)
- D. SELECT customer_id FROM customers HAVING COUNT(orders.customer_id) = 0

<details>
<summary>Show answer</summary>

**B. SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)**

NOT IN against the set of customer_ids that appear in orders returns those with no orders (assuming no NULL customer_id in orders, which would otherwise make NOT IN return no rows). Option A and C both return customers who HAVE ordered, and option D is not valid syntax for this purpose.

</details>

**Q13.** A query reads: SELECT DISTINCT city FROM customers ORDER BY city. The customers table has 500 rows spread across 12 distinct cities. How many rows does this query return? _(basic)_

- A. 500
- B. 12
- C. 1
- D. Depends on the ORDER BY

<details>
<summary>Show answer</summary>

**B. 12**

DISTINCT collapses the result to unique city values, of which there are 12. ORDER BY only affects sort order, not the row count.

</details>

**Q14.** During data cleaning of a products table, you want to find rows where the 'size' column has no value. Which WHERE condition is correct?  ⭐ _(basic)_

- A. WHERE size = NULL
- B. WHERE size IS NULL
- C. WHERE size == NULL
- D. WHERE size <> ''

<details>
<summary>Show answer</summary>

**B. WHERE size IS NULL**

NULL must be tested with IS NULL / IS NOT NULL; comparisons like = NULL or == NULL evaluate to UNKNOWN and never match. Note: size <> '' would only catch non-empty strings, not NULLs.

</details>

**Q15.** Which set of clauses, when combined, returns the top 3 best-selling product categories by total revenue (orders table has category and amount)?  ⭐ _(intermediate)_

- A. SELECT category, SUM(amount) AS rev FROM orders GROUP BY category ORDER BY rev DESC LIMIT 3
- B. SELECT category, SUM(amount) AS rev FROM orders ORDER BY rev DESC GROUP BY category LIMIT 3
- C. SELECT category, SUM(amount) AS rev FROM orders WHERE rev > 0 GROUP BY category LIMIT 3
- D. SELECT TOP 3 category, SUM(amount) FROM orders ORDER BY amount DESC

<details>
<summary>Show answer</summary>

**A. SELECT category, SUM(amount) AS rev FROM orders GROUP BY category ORDER BY rev DESC LIMIT 3**

You must GROUP BY category to aggregate revenue, ORDER BY the summed revenue descending, then LIMIT 3. GROUP BY must come before ORDER BY syntactically, and you cannot reference the aggregate alias in WHERE.

</details>

**Q16.** What does removing duplicate rows from a result set require, and which keyword does it? _(basic)_

- A. UNIQUE applied in the SELECT list
- B. DISTINCT applied after SELECT
- C. DEDUPE() function
- D. GROUP BY is the only way to remove duplicates

<details>
<summary>Show answer</summary>

**B. DISTINCT applied after SELECT**

SELECT DISTINCT removes duplicate rows from the output. UNIQUE is a table constraint, not a SELECT-list keyword, and there is no standard DEDUPE() function. GROUP BY can dedupe but is not the only way.

</details>

### 4.9 Applied Retail / Apparel ML Scenarios

> Which algorithm/approach fits a given retail problem: demand/sales forecasting, product recommendation, customer churn & RFM segmentation, inventory optimization, dynamic pricing, market-basket/association rules, computer vision for apparel, A/B testing. Scenario-style "what would you use" MCQs.
>
> **16 questions**, 13 ⭐ high-yield.

**Q1.** Apparel Group wants to forecast weekly sales for each store-SKU combination, capturing trend, yearly seasonality (Ramadan/Eid spikes, end-of-season sales), and the effect of promotions and price. Which approach is the most appropriate fit?  ⭐ _(basic)_

- A. K-means clustering on the sales values
- B. A supervised time-series / regression model (e.g., gradient-boosted trees or a forecasting model) with calendar, promo and price features
- C. Apriori association-rule mining
- D. Logistic regression on the raw daily timestamps

<details>
<summary>Show answer</summary>

**B. A supervised time-series / regression model (e.g., gradient-boosted trees or a forecasting model) with calendar, promo and price features**

Demand forecasting predicts a continuous quantity over time, so it is a supervised regression/time-series problem; seasonality, promo and price enter as engineered features. K-means and Apriori are unsupervised and do not predict a numeric target; logistic regression is for classification.

</details>

**Q2.** Apparel Group is launching a brand-new e-commerce site. There is almost no user purchase or click history yet, but every product has a rich text description and detailed attributes (fabric, fit, color, category). Which recommendation approach works best at launch?  ⭐ _(intermediate)_

- A. User-based collaborative filtering
- B. Item-based collaborative filtering on the rating matrix
- C. Content-based filtering using product attributes/descriptions
- D. Matrix factorization (ALS) on implicit feedback

<details>
<summary>Show answer</summary>

**C. Content-based filtering using product attributes/descriptions**

With no interaction history, collaborative filtering and matrix factorization suffer the cold-start problem (no co-occurrence signal). Content-based filtering uses item attributes/descriptions, so it can recommend from day one before behavioral data accumulates.

</details>

**Q3.** A retailer wants to segment its customer base into groups using Recency, Frequency, and Monetary (RFM) features, without any predefined labels. Which technique is the standard fit?  ⭐ _(basic)_

- A. K-means clustering
- B. Random forest classification
- C. Linear regression
- D. Apriori algorithm

<details>
<summary>Show answer</summary>

**A. K-means clustering**

RFM segmentation has no target label, making it an unsupervised problem; K-means clusters customers on the three numeric RFM dimensions. Random forest and linear regression are supervised; Apriori finds item co-occurrence rules, not customer segments.

</details>

**Q4.** Predicting whether each customer will churn (stop purchasing) in the next 90 days, given labeled historical examples of churned vs retained customers, is best framed as which type of problem?  ⭐ _(basic)_

- A. Unsupervised clustering
- B. Binary classification
- C. Association rule mining
- D. Dimensionality reduction

<details>
<summary>Show answer</summary>

**B. Binary classification**

There is a labeled binary outcome (churn / no-churn), so this is supervised binary classification (e.g., logistic regression, gradient-boosted trees). Clustering and association mining are unsupervised and produce no churn probability.

</details>

**Q5.** In market-basket analysis on apparel transactions, a rule {belt} -> {dress shoes} has confidence 0.8, but 75% of all baskets contain dress shoes anyway. Which metric should you trust to judge whether the rule is genuinely interesting?  ⭐ _(intermediate)_

- A. Support
- B. Confidence
- C. Lift
- D. Recall

<details>
<summary>Show answer</summary>

**C. Lift**

High confidence can be an artifact of a very popular consequent. Lift = confidence / support(consequent) = 0.8 / 0.75 ≈ 1.07, only slightly above 1, showing the association is weak despite high confidence. (Lift = 1 means independence.)

</details>

**Q6.** Which algorithm is specifically designed to discover frequent itemsets and association rules (items bought together) from retail transaction data?  ⭐ _(basic)_

- A. Apriori
- B. DBSCAN
- C. Gradient boosting
- D. Naive Bayes

<details>
<summary>Show answer</summary>

**A. Apriori**

Apriori mines frequent itemsets using the property that all subsets of a frequent itemset are also frequent, then derives association rules. DBSCAN is density clustering; gradient boosting and Naive Bayes are supervised predictors.

</details>

**Q7.** Apparel Group wants to automatically classify product images into categories (dress, shirt, trousers, shoes) and detect the garment type from user-uploaded photos. Which model family is most appropriate?  ⭐ _(basic)_

- A. ARIMA time-series model
- B. Convolutional Neural Network (CNN)
- C. K-means on raw pixels
- D. Apriori association rules

<details>
<summary>Show answer</summary>

**B. Convolutional Neural Network (CNN)**

Image classification of apparel is the canonical use case for CNNs (e.g., the Fashion-MNIST benchmark), which learn spatial features from pixels. ARIMA is for time series, K-means on raw pixels does not classify, and Apriori is for transaction rules.

</details>

**Q8.** A churn model is trained on data where only 4% of customers churn. The model reports 96% accuracy. Why is accuracy a misleading metric here, and what should be used instead?  ⭐ _(intermediate)_

- A. Accuracy is fine; 96% is excellent
- B. The classes are imbalanced; a model predicting 'no churn' for everyone gets 96% — use precision/recall, F1 or AUC-PR instead
- C. Use mean squared error instead of accuracy
- D. Switch to R-squared as the evaluation metric

<details>
<summary>Show answer</summary>

**B. The classes are imbalanced; a model predicting 'no churn' for everyone gets 96% — use precision/recall, F1 or AUC-PR instead**

With a 4% positive rate, a trivial all-negative classifier scores 96% accuracy yet catches zero churners. Precision, recall, F1 and PR-AUC reflect performance on the rare churn class. MSE and R-squared are regression metrics, not applicable to classification.

</details>

**Q9.** The retailer has rich historical purchase logs (user x item interactions) and wants personalized recommendations that capture latent taste factors and items 'similar users' bought. Which approach fits best?  ⭐ _(intermediate)_

- A. Content-based filtering only
- B. Collaborative filtering / matrix factorization on the user-item interaction matrix
- C. ARIMA forecasting
- D. DBSCAN clustering of images

<details>
<summary>Show answer</summary>

**B. Collaborative filtering / matrix factorization on the user-item interaction matrix**

When abundant interaction history exists, collaborative filtering (e.g., matrix factorization / ALS) learns latent user and item factors and leverages similar-user behavior. Content-based ignores collaborative signal; ARIMA and DBSCAN solve unrelated problems.

</details>

**Q10.** Apparel Group wants to test whether a new 'Buy Now' button color increases checkout conversion. They split web traffic 50/50 between the old and new versions and compare conversion rates. This experimental method is called:  ⭐ _(basic)_

- A. Cross-validation
- B. A/B testing (controlled experiment)
- C. Market basket analysis
- D. Hyperparameter tuning

<details>
<summary>Show answer</summary>

**B. A/B testing (controlled experiment)**

Randomly splitting users between a control (A) and treatment (B) and comparing an outcome metric is A/B testing. Cross-validation and hyperparameter tuning are model-building techniques; market basket analysis finds item co-occurrence.

</details>

**Q11.** In the A/B test above, version B shows a 0.4 percentage-point higher conversion than A, but the result has a p-value of 0.32. What is the correct interpretation?  ⭐ _(intermediate)_

- A. B is definitively better; roll it out immediately
- B. The lift is not statistically significant; the difference could be due to chance, so do not conclude B is better yet
- C. A p-value of 0.32 means B is 32% better
- D. The test is invalid and must be discarded

<details>
<summary>Show answer</summary>

**B. The lift is not statistically significant; the difference could be due to chance, so do not conclude B is better yet**

A p-value of 0.32 (well above the usual 0.05 threshold) means that, if there were truly no difference, an effect this large or larger would occur 32% of the time by chance — so the result is not statistically significant. A p-value is not a percentage improvement.

</details>

**Q12.** A retailer wants to set the optimal selling price for each apparel item to maximize revenue, learning how demand responds to price and adjusting prices in near real time. This problem is best described as: _(intermediate)_

- A. Dynamic pricing, often modeled with demand/price-elasticity estimation or reinforcement learning
- B. Market basket analysis with Apriori
- C. Image segmentation with a CNN
- D. RFM clustering with K-means

<details>
<summary>Show answer</summary>

**A. Dynamic pricing, often modeled with demand/price-elasticity estimation or reinforcement learning**

Optimizing price against price-sensitive demand is dynamic pricing, typically built on price-elasticity/demand models or reinforcement learning that adapts to feedback. The other techniques address item co-occurrence, vision, and segmentation respectively.

</details>

**Q13.** For a fashion search feature, the team needs to find visually similar garments to a query image (e.g., 'find dresses that look like this'). Which approach is most suitable? _(intermediate)_

- A. Compute image embeddings with a CNN and retrieve nearest neighbors in embedding space
- B. Run Apriori on the product catalog
- C. Fit an ARIMA model on the image pixels
- D. Apply logistic regression to the raw RGB values

<details>
<summary>Show answer</summary>

**A. Compute image embeddings with a CNN and retrieve nearest neighbors in embedding space**

Visual similarity search embeds images into a vector space using a CNN, then retrieves nearest neighbors by distance/cosine similarity. Apriori, ARIMA, and raw-pixel logistic regression are not designed for visual similarity retrieval.

</details>

**Q14.** An inventory team must decide reorder quantities that balance stockout risk against holding cost, given uncertain demand. Which framing best matches this problem? _(advanced)_

- A. Forecast demand (with uncertainty) then apply an inventory-optimization model such as a (s, S) / newsvendor / safety-stock policy
- B. Cluster the SKUs with K-means and reorder the largest cluster
- C. Mine association rules between SKUs
- D. Train a CNN on the warehouse photos

<details>
<summary>Show answer</summary>

**A. Forecast demand (with uncertainty) then apply an inventory-optimization model such as a (s, S) / newsvendor / safety-stock policy**

Inventory optimization combines a probabilistic demand forecast with an optimization/policy step (newsvendor, safety stock, reorder points) that trades off stockout vs holding cost. Clustering, association mining, and image models do not determine reorder quantities.

</details>

**Q15.** Select ALL techniques that are UNSUPERVISED (require no labeled target) as typically applied in retail analytics.  ⭐ _(intermediate)_

- A. K-means RFM customer segmentation
- B. Apriori market-basket association rules
- C. XGBoost weekly sales-forecast regression
- D. Logistic regression churn classification

<details>
<summary>Show answer</summary>

**A. K-means RFM customer segmentation, B. Apriori market-basket association rules**

K-means clustering and Apriori association-rule mining learn structure from unlabeled data. XGBoost sales forecasting (regression) and logistic-regression churn prediction (classification) both require labeled targets and are supervised.

</details>

**Q16.** A demand-forecasting model fits the training sales data almost perfectly but performs poorly on the held-out validation weeks. What is this called, and a reasonable remedy?  ⭐ _(intermediate)_

- A. Underfitting; add more model complexity
- B. Overfitting; apply regularization, pruning, or reduce model complexity / add more data
- C. Data leakage; it cannot be fixed
- D. Class imbalance; apply SMOTE

<details>
<summary>Show answer</summary>

**B. Overfitting; apply regularization, pruning, or reduce model complexity / add more data**

Excellent training but poor validation performance is the signature of overfitting (high variance). Remedies include regularization, tree pruning, simpler models, or more data. SMOTE addresses class imbalance, not forecasting variance.

</details>

### 4.10 Hands-on Coding (Python)

> Short coding-simulator / pseudo-code problems typical of these tests: array/string manipulation, dict/counting, a small pandas transformation, implement a metric from scratch (accuracy, euclidean distance, normalize), simple algorithm (two-sum, max subarray). Provide the full reference solution as a fenced Python code block.
>
> **16 questions**, 12 ⭐ high-yield.

**Q1.** What is the output of the following code?

```python
counts = {}
for c in 'apparel':
    counts[c] = counts.get(c, 0) + 1
print(counts['a'], counts['p'])
```  ⭐ _(basic)_

- A. 2 1
- B. 1 2
- C. 2 2
- D. KeyError

<details>
<summary>Show answer</summary>

**C. 2 2**

'apparel' spells a-p-p-a-r-e-l: 'a' appears at indices 0 and 3 (count 2), 'p' at indices 1 and 2 (count 2). dict.get(c, 0) safely returns 0 for unseen keys, so no KeyError occurs.

</details>

**Q2.** Which expression returns the SKU code 'SHIRT' (the last 5 characters) from the string s = 'AG-2026-SHIRT'?  ⭐ _(basic)_

- A. s[-5:]
- B. s[5:]
- C. s[:-5]
- D. s[5:-1]

<details>
<summary>Show answer</summary>

**A. s[-5:]**

Negative slice s[-5:] takes the last 5 characters ('SHIRT'). s[:-5] drops the last 5 instead ('AG-2026-'), and s[5:] starts at index 5 giving '26-SHIRT' (index 5 is the second '2').

</details>

**Q3.** What does this code print?

```python
prices = [199, 499, 299]
print([p * 2 for p in prices if p < 400])
```  ⭐ _(basic)_

- A. [398, 598]
- B. [398, 998, 598]
- C. [199, 299]
- D. [398, 598, 998]

<details>
<summary>Show answer</summary>

**A. [398, 598]**

The filter keeps only 199 and 299 (both < 400), then doubles them to 398 and 598. 499 is excluded by the condition.

</details>

**Q4.** What is the output?

```python
def discount(price, pct=10):
    return price - price * pct / 100
print(discount(200))
``` _(basic)_

- A. 180.0
- B. 190.0
- C. 200
- D. 180

<details>
<summary>Show answer</summary>

**A. 180.0**

pct defaults to 10, so 200 - 200*10/100 = 200 - 20 = 180. The / operator produces a float in Python 3, so the result prints as 180.0, not 180.

</details>

**Q5.** Implement the classic two-sum: return the indices of the two numbers in the list that add up to target. Assume exactly one solution exists.  ⭐ _(intermediate)_

<details>
<summary>Show answer</summary>

**```python
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
    return []

# Example: two_sum([2, 7, 11, 15], 9) -> [0, 1]
```**

A single pass with a hash map gives O(n) time and O(n) space. For each element store value->index; if the complement (target - n) was already seen, return both indices. The brute-force double loop is O(n^2). Verified: two_sum([2,7,11,15],9) -> [0, 1].

</details>

**Q6.** What is the output of this code?

```python
try:
    qty = [10, 20, 30]
    print(qty[3])
except IndexError:
    print('out of range')
except ZeroDivisionError:
    print('div zero')
```  ⭐ _(basic)_

- A. out of range
- B. div zero
- C. 30
- D. None

<details>
<summary>Show answer</summary>

**A. out of range**

qty[3] is out of bounds for a 3-element list (valid indices 0-2), raising IndexError, which the first matching except block catches and prints 'out of range'.

</details>

**Q7.** Given the DataFrame df with columns 'category' and 'sales', which line returns total sales per category as a Series?  ⭐ _(intermediate)_

- A. df.groupby('category')['sales'].sum()
- B. df['sales'].groupby().sum('category')
- C. df.sum('sales').groupby('category')
- D. df.groupby('sales')['category'].sum()

<details>
<summary>Show answer</summary>

**A. df.groupby('category')['sales'].sum()**

groupby('category') splits rows by category, ['sales'] selects the column to aggregate, and .sum() combines each group, returning a Series indexed by category. Option D groups by the wrong column, and the others use invalid syntax.

</details>

**Q8.** Which pandas expression keeps only the rows where the 'price' column is greater than 1000?  ⭐ _(intermediate)_

- A. df[df['price'] > 1000]
- B. df.filter('price' > 1000)
- C. df[df > 1000]['price']
- D. df.where('price' > 1000)

<details>
<summary>Show answer</summary>

**A. df[df['price'] > 1000]**

Boolean masking df[df['price'] > 1000] returns the rows where the condition is True. df.filter is for selecting labels (not value conditions), and 'price' > 1000 alone compares a string to an int (a TypeError).

</details>

**Q9.** Implement accuracy from scratch (no sklearn): given two equal-length lists y_true and y_pred, return the fraction of matching predictions.  ⭐ _(intermediate)_

<details>
<summary>Show answer</summary>

**```python
def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

# Example: accuracy([1, 0, 1, 1], [1, 1, 1, 0]) -> 0.5
```**

Accuracy = (number of correct predictions) / (total predictions). zip pairs each true label with its prediction; summing the matches and dividing by the count gives the fraction. Guarding the empty case avoids ZeroDivisionError. Verified: example returns 0.5 (2 of 4 match).

</details>

**Q10.** Implement the Euclidean distance between two equal-length numeric vectors a and b, without using numpy.  ⭐ _(intermediate)_

<details>
<summary>Show answer</summary>

**```python
import math

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# Example: euclidean([0, 0], [3, 4]) -> 5.0
```**

Euclidean distance is the square root of the sum of squared coordinate differences. zip pairs the components; the generator squares each difference, sum aggregates, and math.sqrt takes the root. For [0,0] and [3,4] this gives sqrt(9+16)=5.0.

</details>

**Q11.** What does the following function compute?

```python
def f(lst):
    m = lst[0]
    for x in lst:
        if x > m:
            m = x
    return m
``` _(basic)_

- A. The maximum element of lst
- B. The minimum element of lst
- C. The sum of lst
- D. The last element of lst

<details>
<summary>Show answer</summary>

**A. The maximum element of lst**

It initializes m to the first element and replaces m whenever a larger element is found, so it returns the maximum. It would compute the minimum only if the comparison were x < m.

</details>

**Q12.** Implement min-max normalization: scale a numeric list to the range [0, 1] using (x - min) / (max - min). Handle the case where all values are equal.  ⭐ _(intermediate)_

<details>
<summary>Show answer</summary>

**```python
def normalize(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]

# Example: normalize([10, 20, 30]) -> [0.0, 0.5, 1.0]
```**

Min-max scaling maps the minimum to 0 and the maximum to 1 linearly. The hi == lo guard avoids division by zero when every value is identical (a constant feature), returning zeros instead of crashing. Verified: normalize([10,20,30]) -> [0.0, 0.5, 1.0].

</details>

**Q13.** Solve the maximum subarray sum (Kadane's algorithm): return the largest sum of any contiguous subarray of nums.  ⭐ _(intermediate)_

<details>
<summary>Show answer</summary>

**```python
def max_subarray(nums):
    best = cur = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best

# Example: max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> 6
```**

Kadane's runs in O(n): cur tracks the best subarray sum ending at the current index (extend or restart), and best records the global maximum. For the example, subarray [4,-1,2,1] sums to 6.

</details>

**Q14.** What is the output?

```python
from collections import Counter
sizes = ['S', 'M', 'M', 'L', 'M', 'S']
print(Counter(sizes).most_common(1))
```  ⭐ _(intermediate)_

- A. [('M', 3)]
- B. ('M', 3)
- C. [('M', 2)]
- D. {'M': 3}

<details>
<summary>Show answer</summary>

**A. [('M', 3)]**

Counter tallies frequencies ('M' appears 3 times). most_common(1) returns a list of the single most frequent (key, count) tuple, hence [('M', 3)].

</details>

**Q15.** What does this code print?

```python
s = 'level'
print(s == s[::-1])
``` _(basic)_

- A. True
- B. False
- C. 'level'
- D. None

<details>
<summary>Show answer</summary>

**A. True**

s[::-1] reverses the string. 'level' reversed is still 'level', so the equality is True — this is the standard one-line palindrome check.

</details>

**Q16.** Which statement(s) about the line below are TRUE? (Select all that apply.)

```python
result = df.groupby('store')['revenue'].mean()
``` _(advanced)_

- A. result is a pandas Series indexed by store
- B. It computes the mean revenue for each store
- C. It modifies df in place
- D. Stores with no revenue rows are excluded from the result

<details>
<summary>Show answer</summary>

**A. result is a pandas Series indexed by store; B. It computes the mean revenue for each store; D. Stores with no revenue rows are excluded from the result**

groupby on one key and aggregating one column yields a Series indexed by that key, with the per-group mean; group keys arise only from rows present in df, so a store with no rows never appears (D true). groupby does not mutate df in place (C false).

</details>

---

## 5. Final-hour checklist

- **Format:** ~18–25 questions, <45 min → budget ~90–120s each; never sink 5 min into one item — flag and move on.
- **Likely proctored:** webcam + full-screen lock + tab-switch/copy-paste detection are common on Mettl/iMocha. Quiet, well-lit room; close other tabs/apps; don’t leave full-screen.
- **Environment:** updated Chrome/Edge on a laptop (not phone), stable internet, charger plugged in, 45 min uninterrupted.
- **Rusty-recall traps to refresh:** exact precision/recall/F1 formulas; which algos need feature scaling (KNN/SVM/PCA yes, trees no); L1→sparsity vs L2→shrinkage; sigmoid 0–1 / tanh −1–1 / ReLU 0–∞ / softmax sums to 1; `fit_transform` on train only, `transform` on test; bagging reduces variance vs boosting reduces bias.
- **MCQ technique:** read every option before answering; eliminate obviously wrong ones; watch for “NOT/EXCEPT” wording.
- **Negative marking is unknown** — don’t assume it exists; with limited time, an educated guess on a flagged question usually beats leaving it blank, but confirm on the instructions screen if shown.

---

## Sources

Portal templates and interview banks cross-referenced to reconstruct this bank (verify model names/benchmarks before quoting — fast-moving space):

- gyansetu.in ML MCQ (Q15/Q17); imocha Regression Analysis test
- gyansetu.in ML MCQ (Q14/Q17)
- gyansetu.in ML MCQ (Q54)
- gyansetu.in ML MCQ (Q55)
- mettl.com ML Engineer assessment; gyansetu.in (Q20/Q57/Q58)
- gyansetu.in ML MCQ (Q59); imocha ML test
- gyansetu.in ML MCQ (Q57); mettl.com assessment
- imocha ML test; mettl.com ML Engineer assessment
- imocha ML test; sanfoundry SVM MCQ
- gyansetu.in ML MCQ (Q51); sanfoundry SVM MCQ
- sanfoundry SVM MCQ; imocha ML test
- gyansetu.in ML MCQ (Q12/Q52); sanfoundry KNN MCQ
- gyansetu.in ML MCQ (Q22); sanfoundry clustering MCQ
- gyansetu.in ML MCQ (Q24)
- gyansetu.in ML MCQ (Q53); sanfoundry Naive Bayes MCQ
- gyansetu.in ML MCQ (Q7); imocha Mathematics for ML test
- gyansetu.in ML MCQ (Q70); mettl.com & imocha gradient-descent items
- mettl.com ML Engineer assessment; imocha gradient-descent items
- mettl.com & imocha gradient-descent items
- Adaface NLP Online Test; AnalyticsVidhya text vectorization guide
- Adaface NLP Online Test; AnalyticsVidhya
- Adaface NLP Online Test; Turing word embeddings guide
- GeeksforGeeks Word Embeddings; Turing
- TowardsAI 40 GenAI Interview Questions 2026; InterviewBit LLM
- Attention Is All You Need; DataCamp LLM questions
- InterviewBit LLM 2026; DataCamp
- Adaface NLP Online Test; InterviewBit
- TowardsAI 40 GenAI Interview Questions 2026; DataCamp
- TowardsAI 40 GenAI Interview Questions 2026; DataCamp RAG vs fine-tune
- DataCamp; TowardsAI 2026
- DataCamp; TowardsAI vector DB / hybrid search
- Adaface NLP Online Test (WSD via cosine similarity); GeeksforGeeks
- InterviewBit LLM 2026; TowardsAI 2026
- https://mettl.com/en/test/machine-learning-engineer-python-assessment/
- https://mettl.com/en/test/machine-learning-engineer-assessment/
- https://www.imocha.io/tests/machine-learning-with-python-test
- https://www.adaface.com/assessment-test/machine-learning-online-test
- https://www.testdome.com/tests/machine-learning-online-test/267
- https://www.glassdoor.com/Interview/Apparel-Group-Interview-Questions-E534323.htm

---

_Compiled for Sachin Singh. Representative practice reconstructed from public assessment-platform templates; not affiliated with Apparel Group and not actual exam questions._
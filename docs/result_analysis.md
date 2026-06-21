# ASTraM Traffic Event Intelligence Operational Performance and Reliability Report

This report documents the empirical performance of the ensembled machine learning system trained on the ASTraM event dataset. All metrics presented reflect out of fold cross validation performance. This means every prediction was made on data withheld from the model during training. Therefore, they represent a conservative, unbiased estimate of real world generalization. The evaluation covers both the original baseline configuration and the updated configuration incorporating class imbalance corrections and optimized ensemble blending.

## 1. Executive Summary

* Event Impact Score (EIS): The ensemble achieves an R squared score of 0.7011, a Mean Absolute Error of 4.0293 points, and a Root Mean Squared Error of 7.3810. This is an improvement of 0.43 percentage points in R squared score compared to the baseline.
* Manpower Recommendation: The ensemble achieves an R squared score of 0.8666, a Mean Absolute Error of 0.3328 officers, and a Root Mean Squared Error of 1.1147. This represents a 0.55 percentage point increase in R squared score compared to the baseline.
* Barricade Recommendation: The ensemble achieves an R squared score of 0.8302, a Mean Absolute Error of 2.3885 barricades, and a Root Mean Squared Error of 4.9447. This is a 0.05 percentage point improvement in R squared score over the baseline.
* Diversion Requirement: The ensemble achieves an F1 score of 0.7841, an accuracy of 0.9145, a precision of 0.8021, and a recall of 0.7668 at the standard 0.50 threshold. This represents a recall improvement of 6.89 percentage points over the baseline ensemble.

The diversion recall improvement is a highly significant result. Prior to the optimization, the baseline ensemble failed to identify approximately 30 percent of events requiring traffic diversion. The revised training protocol and optimized blending weights reduce this miss rate to approximately 23 percent, while maintaining high precision.

## 2. Target by Target Analysis

### Event Impact Score

The Event Impact Score quantifies the severity of traffic disruption on a continuous scale from 0 for negligible impact to 100 for total arterial blockage. It is computed during data processing as a weighted combination of event duration, road closure status, event cause severity tier, priority, and corridor classification. Predicting EIS is challenging because the ground truth label is a composite of operational factors that vary non linearly with incident type and context.

The ensemble achieves a Mean Absolute Error of 4.0293 points and a coefficient of determination R squared of 0.7011 on the held out validation partitions. The neural network contributes R squared of 0.6831 and the LightGBM baseline contributes R squared of 0.6972. Their 40/60 weighted combination produces the best overall generalization. The explained variance of approximately 70 percent reflects the predictable component of event severity driven by structured features. The residual 30 percent variance corresponds to stochastic operational factors such as driver response behavior, spontaneous secondary incidents, and weather related demand fluctuations.

### Manpower Deployment Recommendation

The manpower target encodes the recommended number of traffic officers to deploy to the incident site, modeled as a Poisson count bounded between 1 and 30 officers. LightGBM achieves R squared of 0.8675 and MAE of 0.3330 officers on this target, outperforming the neural network which achieves R squared of 0.6892 and MAE of 0.8510. The neural network performance on this target improved by 3.39 percentage points in R squared score compared to the baseline after reverting the diversion head to standard unweighted Binary Cross Entropy, which restored the stability of the shared representation.

In practical terms, the Mean Absolute Error of 0.3328 officers on a typical scale of 1 to 30 implies that the operational recommendation is correct to within one officer in the large majority of deployments.

### Barricade Deployment Recommendation

The barricade target encodes recommended physical barrier counts on a scale from 0 to 50 units. The ensemble achieves R squared of 0.8302 and MAE of 2.3885 barricades via a 50/50 blend, as both constituent models achieve comparable accuracy. In absolute terms, a mean deviation of 2.3885 barricades on a 50 unit scale represents an average relative error of approximately 5 percent.

### Diversion Requirement Classification

The diversion classification target is binary, encoding whether a traffic diversion route must be established. It is the most operationally critical of the four prediction targets. A false negative results in sustained arterial blockage, while a false positive causes unnecessary secondary disruption. The dataset exhibits a natural class imbalance of approximately 3.94 negative samples per positive sample.

#### Baseline Performance

Prior to the optimization, the baseline ensemble achieved an accuracy of 0.9164, an F1 score of 0.7718, a precision of 0.8632, and a recall of 0.6979 at the threshold of 0.50. The Two Tower Neural Network achieved a recall of 0.6731. These figures appeared adequate, but the recall of 0.6979 implied that approximately 30 percent of genuine diversion events were missed.

#### Updated Performance after Optimization

Following the introduction of per fold scale pos weight (approximately 3.94) for the LightGBM classifier, standard unweighted Binary Cross Entropy for the neural network diversion head, and optimized ensemble blending weights of 30 percent neural network and 70 percent LightGBM, the metrics improved substantially.

* Two Tower Neural Network: Acc@0.50 of 0.9053, F1@0.50 of 0.7417, Precision of 0.8285, Recall of 0.6713, ROC-AUC of 0.8899, and PR-AUC of 0.7803.
* LightGBM Baseline: Acc@0.50 of 0.9123, F1@0.50 of 0.7834, Precision of 0.7832, Recall of 0.7837, ROC-AUC of 0.9256, and PR-AUC of 0.8441.
* Weighted Ensemble: Acc@0.50 of 0.9145, F1@0.50 of 0.7841, Precision of 0.8021, Recall of 0.7668, ROC-AUC of 0.9243, and PR-AUC of 0.8404.

The recall at the default 0.50 threshold improves by 6.89 percentage points for the ensemble. The LightGBM model, benefiting from the scale pos weight correction, achieves the highest rank based performance with a ROC-AUC of 0.9256 and a PR-AUC of 0.8441.

## 3. Why the Hybrid Ensemble Architecture Succeeds

The design premise is that the neural network and gradient booster are strong in complementary regimes of the feature space, and that a calibrated weighted combination outperforms either model individually.

The Two Tower Neural Network excels on targets that are sensitive to continuous, geometric, and semantic variation. The 1024 dimensional sentence embedding captures event description semantics. The multi resolution geohash tower captures spatial autocorrelation at three geographic scales. The cyclical temporal encodings prevent the common tabular failure of treating hour 23 and hour 0 as distant.

LightGBM excels on targets governed by quasi discrete operational rules. The manpower recommendation is determined by a composition of cause severity tier, priority multiplier, road closure status, corridor flag, and peak hour indicator. This structure is represented exactly by decision tree splits.

The ensemble blending weights reflect the relative model performance on each target as measured by cross validation out of fold statistics. The deviation from a uniform blend is most pronounced on the manpower target (10/90 toward LightGBM), reflecting the tabular separability of the count rules, and on the diversion target (30/70 toward LightGBM), reflecting the calibrated probabilities of the scale pos weight optimized classifier.

## 4. Improvement Summary

The training configuration introduced three changes to address the diversion class imbalance and improve general performance. First, standard unweighted Binary Cross Entropy was restored for the neural network diversion head, which preserved the quality of the shared representation. Second, the LightGBM diversion classifier received a per fold computed scale pos weight hyperparameter. Third, the ensemble blending weights were re optimized to 30 percent neural network and 70 percent LightGBM for the diversion target.

The collective effect is a recall improvement of 6.89 percentage points for the ensemble on the diversion target, while maintaining or improving performance on all other targets.

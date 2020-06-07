# Automated Issue Assignment: Results and Insights from an Industrial Case

This project is about how we automated the process of assigning issue reports to development teams by using data mining approaches in an industrial setting and our experience gained by deploying the resulting system, IssueTAG. (Note that the related paper is currently under review.)

Due to security reasons, we are able to publish (in full, or in partial) neither the issue reports used in these studies nor the source code of IssueTAG.  However, some scripts, which are similar to the ones we used in order to carry out the experiments for the issue assignment approach and time locality of training data, as well as some code excerpts demonstrating the basic functionalities used in explaining the assignments and monitoring deterioration, can be found here. 

## IssueTAG

IssueTAG is a software issue report assignment tool, which has been fully operational in an industrial setting since Jan 12, 2018, making automated assignments for all the issue reports submitted. Our goal in this work is to identify an existing approach that can produce similar assignment accuracies with manual assignment and that can be developed and deployed with as little risk as possible.

### Issue Assignment Approach

Given an issue report, we first combine the 'description' and 'summary' attributes of the issue report, tokenize the combined text into terms, and remove the non-letter characters as well as the stop words. We then represent an issue report as a multi-dimensional vector using the well-known 'tf-idf' method. Finally, the problem of assignment is cast to a classification problem where the development team, to which the issue report should be assigned, becomes the class to be predicted.

```
[compare_algorithms.ipynb](https://github.com/ethemutku/IssueTAG/blob/master/compare_algoritms.ipynb) is the script for comparing different machine learning algorithms for issue report classification.
```

### Time Locality of Training Data

IssueTAG is an online system, which is expected to have a long lifespan. Therefore, the classification model it uses for making the assignments should be trained as needed since the underlying issue database evolves.

To this end, we used the 'sliding window' and 'cumulative window' approaches introduced in ([Jonsson et al. 2016](https://www.researchgate.net/publication/281740475_Automated_Bug_Assignment_Ensemble-based_Machine_Learning_in_Large_Scale_Industrial_Contexts)).

```
[evaluate_time_and_amount.ipynb](https://github.com/ethemutku/IssueTAG/blob/master/evaluate_time_and_amount.ipynb) is for the evaluation of the effect of varying time interval and amount of training data.
```

## Explaining Team Assignments

One interesting observation we made after IssueTAG had been deployed was that, occasionally, especially for incorrect assignments, the stakeholders demanded some explanations as to why and how certain issue reports had been assigned to their teams. This was an issue we didn't expect to face before deploying the system. As a matter of fact, based on the informal discussions we had with the stakeholders, we quickly realized that explaining the assignments could further improve the trust in IssueTAG.

We use LIME (Local Interpretable Model-Agnostic Explanations) to automatically produce explanations for the issue assignments made by IssueTAG. LIME is a model-agnostic algorithm for explaining the predictions of a classification or regression model ([Riberio et al. 2016](https://github.com/marcotcr/lime)).

```
[explain_issues.ipynb](https://github.com/ethemutku/IssueTAG/blob/master/explain_issues.ipynb) is the script to explain specific test issue records.
```

## Monitoring Deterioration

A mechanism to monitor the performance of IssueTAG is important because it not only increases the confidence of the stakeholders in the system, but also helps determine when the underlying classification model needs to be recalibrated by, for example, retraining the model.

To this end, we use an online change point detection approach, called Pruned Exact Linear Time (PELT) (Killick et al. 2012). In a nutshell, PELT is a statistical analysis technique to identify when the underlying model of a signal changes (Truong et al. 2018). In our context, we feed PELT with a sequence of daily assignment accuracies as the signal. The output is a set of points in time (if any) where mean shifts. PELT, being an approach based on dynamic programming, detects both the number of change points and their locations with a linear computational cost under certain conditions (Killick et al. 2012). 

```
[monitor_performance.ipynb](https://github.com/ethemutku/IssueTAG/blob/master/monitor_performance.ipynb) is the script to monitor the performance of the issue assignment system.
```

## Lessons Learnt

It is not just about deploying a system for automated issue assignment, but also about designing/changing the assignment process around the system. 

The accuracy of the assignments does not have to be higher than that of manual assignments in order for the system to be useful. 

Deploying such a system requires the development of additional functionalities, such as creating human-readable explanations for the assignments and detecting deteriorations in assignment accuracies, for both of which we have developed and empirically evaluated different approaches. 

Stakeholders do not necessarily resist change and gradual transition helps build confidence.

## References

[1] Jonsson, L., Borg, M., Broman, D., Sandahl, K., Eldh, S., & Runeson, P. (2016). Automated bug assignment: Ensemble-based machine learning in large scale industrial contexts. Empirical Software Engineering, 21(4), 1533-1578.

[2] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). "Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

[3] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590-1598.

[4] Truong, C., Oudre, L., & Vayatis, N. (2018). ruptures: change point detection in Python. arXiv preprint arXiv:1801.00826.


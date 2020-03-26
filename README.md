This project includes the analysis scripts for the replication of the paper "Automated Issue Assignment: Results and Insights from an Industrial Case".

algorithm_comparison.py is the script for comparing different machine learning algorithms for issue report classification (Appendix A: Evaluating Existing Issue Assignment Approaches).

time_based_evaluation.py is the script for the evaluation of the results with varying time interval and amount of data  (Appendix B: Time Locality and Amount of Training Data).

Note that before running the scripts TextPreProcessor.py and DataLoader.py files need to be changed, both of which include code which may vary depending on different issue report repositories and on different evaluations.  

This project includes the analysis scripts for the replication of the paper "Automated Issue Assignment: Results and Insights from an Industrial Case".

monitor_performance.py is the script to monitor the performance of the issue assignment system. (Section 6 Monitoring Deterioration)

train_test.py is the script 

1. for comparing different machine learning algorithms for issue report classification (Appendix A: Evaluating Existing Issue Assignment Approaches).

2. for the evaluation of the results with varying time interval and amount of training data (Appendix B: Time Locality and Amount of Training Data).
    
3. for explaining specific test issue records. (Section 5 Explaining Team Assignments) 

Check these two files according to your data and language specifications:

1. DataLoader.py includes the classes to load and filter your data.

2. TextPreProcessor.py includes the class to preprocess textual data. 

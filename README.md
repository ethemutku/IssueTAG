This project includes the analysis scripts for the replication of the paper "Automated Issue Assignment: Results and Insights from an Industrial Case".

train_test.py is the script 
    1. for comparing different machine learning algorithms for issue report classification (Appendix A: Evaluating Existing Issue Assignment Approaches).
    2. for the evaluation of the results with varying time interval and amount of training data (Appendix B: Time Locality and Amount of Training Data). 
    3. for explaining specific test issue records. (Section 5 Explaining Team Assignments) 

Note that before running the scripts TextPreProcessor.py and DataLoader.py files need to be changed, both of which include code which may vary depending on different issue report repositories and on different evaluations.  

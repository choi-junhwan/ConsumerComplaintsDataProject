# ConsumerComplaintsDataProject
This is my data science side project.
The data Consumer_Complaints.csv comes from http://catalog.data.gov/dataset/consumer-complaint-database.
I wrote a python analysis script, ConsumerComplaints_Analy.py.
This script performs following analyses as follow.
[Outputs are in output.dat file and there are three plots, bar.png, ComplainCount.png, and ComplainCount_WC.png.]

1) It demonstrates the contents and size of the file.
   Main contents, which I use in this analysis are 'Date received','Issue','Product','Timely response?', and 'Consumer disputed?'

2) Compute the Pearson's correlation between 'Timely response?' and 'Consumer disputed?'.
   I find that there is very little correlation, correlation coefficient = 0.0334 and p-value ~ 0.

3) I select "Top 5 most frequently complained products", "Top 5 most frequently complained products which are not Timely responded", and "Top 5 most frequently complained products which are Consumer disputed" and I find "Mortgage" and "Debt collection" are top 2 products.
   "Debt collection" complaint gets less timely response while "Mortgage" complaint gets the highest rate to Consumer disputed.
   I also plot the time evolution of the top 3 complained products year between 2011 to 2015 in bars.png.

4) I select "Top 5 most frequently complained issues", "Top 5 most frequently complained issues which are not Timely responded", and "Top 5 most frequently complained issues which are Consumer disputed" and I find "Loan modification,collection,foreclosure".
   "Cont'd attempts collect debt not owed" complaint, which is 4th most frequently complained issue, gets less timely response.
   "Loan modification,collection,foreclosure" complaint, which is 1st most frequently complained issue, gets the highest rate to Consumer disputed.
   In addition, I plot the time evolution of the total number of complaint with 1st and 2nd most frequently complained issues in ComplainCount.png.

5) Lastly, I perform a topic modeling with top 25 most frequently complained issues using LDA method.
   The output of the LDA analysis are [u'0.076*loan + 0.041*servic + 0.041*payment', u'0.035*money + 0.035*make + 0.035*send', u'0.082*credit + 0.056*report + 0.056*collect', u'0.071*account + 0.071*close + 0.039*problem', u'0.050*ident + 0.050*theft + 0.050*fraud']
   The realted result is also shown in ComplainCount_WC.png as word cloud.
# ConsumerComplaintsDataProject
This is my data science side project.
The data Consumer_Complaints.csv comes from http://catalog.data.gov/dataset/consumer-complaint-database.
I wrote a python analysis script, ConsumerComplaints_Analy.py.
This script performs following analyses as follow.
[Outputs are in output.dat file and there are three plots, bar.png, ComplainCount.png, and ComplainCount_WC.png.]

1) It demonstrates the contents and size of the file.
   Main contents, which I use in this analysis are 'Date received', 'Issue', 'Product', 'Timely response?', and 'Consumer disputed?'

2) Compute the Pearson's correlation between 'Timely response?' and 'Consumer disputed?'
   I find that there is very little correlation, correlation coefficient = 0.0334 and p-value ~ 0.

3) I select "Top 5 most frequently complained products", "Top 5 most frequently complained products which are not Timely responded", and "Top 5 most frequently complained products which are Consumer disputed" and I find "Mortgage" and "Debt collection" are top 2 products.
   "Debt collection" complaint gets less timely response while "Mortgage" complaint gets the highest rate to Consumer disputed.
   I also plot the time evolution of the top 3 complained products year between 2011 and 2015 in bars.png.

4) I select "Top 5 most frequently complained issues", "Top 5 most frequently complained issues which are not Timely responded", and "Top 5 most frequently complained issues which are Consumer disputed" and I find "Loan modification, collection, foreclosure".
   "Cont'd attempts collect debt not owed" complaint, which is 4th most frequently complained issue, gets less timely response.
   "Loan modification, collection, foreclosure" complaint, which is 1st most frequently complained issue, gets the highest rate to Consumer disputed.
   In order to make clear view, I plot annual time evolution of the total number of complaint with top three most frequently complained issues, the rate of NOT Timely responded, and the rate of Consumer disputed.
   I see that the timely responded rate is improved over the last 4 years, while the Consumer disputed rate is more less constant.

5) In order to make different perspective to the topic of the issue, I perform a topic modeling with top 25 most frequently complained issues using LDA method.
   The outputs of the LDA analysis are
   u'0.107*credit + 0.073*report + 0.040*unabl', 
   u'0.038*payment + 0.037*send + 0.037*make', 
   u'0.055*servic + 0.030*debt + 0.030*attempt', 
   u'0.052*problem + 0.052*account + 0.052*close', 
   u'0.038*loan + 0.038*manag + 0.038*collect'
   The topic list is shown the left panel of the ComplainCount_WC.png.
   In addition, the related words for the issues are shown in right panel if the ComplainCount_WC.png as word cloud with popular words shown in larger fonts.

Future, I would like to dissect the response and disputed rate depending on the state to see which states need to be improve the service. I also plan to topic modeling with "consumer complaint narratives" to understand general concerns from the consumers. 
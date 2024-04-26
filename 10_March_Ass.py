#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Qus 1

Estimation statistics involves using sample data to make educated guesses or estimates about population parameters . point estimate and interval estimate are two common techniques used in estimation statistics.

1. Point Estimate:
    A point estimate is a single value that serves as the best guess or estimate of a population parameter. It's typically calculated from sample data and used to approximate an unknown population parameter.
2. Interval Estimate:
    An interval estimate , on the other hand , provides a range of values within which the true population parameter is li,ely to lie , along with a level of confidence associated with the range . It takes into account the variability of the data and provides a more informative estimate than a single point.
# In[2]:


# Qus 2


# In[8]:


def estimate_mean(sample_mean,sample_std, sample_size):
    standard_error=sample_std/(sample_size ** 0.5)
    
    margin_of_error=1.96*standard_error
    
    lower_bound=sample_mean - margin_of_error
    upper_bound=sample_mean + margin_of_error
    
    
    return sample_mean,lower_bound , upper_bound
    
sample_mean=50
sample_std=10
sample_size=100

estimated_mean,lower_bound,upper_bound=estimate_mean(sample_mean,sample_std,sample_size)

print(estimated_mean)
print("95% confidence interval:[{:.2f},{:.2f}]".format(lower_bound,upper_bound))


# In[9]:


# Qus 3

Hypothesis testing is a statistical method used to make infrences about a population based on sample data. it involves evaluting two competing hypotheses about a population parameter. the null hypothesis (Ho) and the alternative hypothesis (Ha).
the null hypothesis typically represents a default assumption or a statement of no effect, while the alternative hypothesis represents the researcher's claim or the effect they are interested in.

The process of hypothesis testing involves the following steps:
    1. Formula Hypotheses: Define the null hypothesis (Ho) and the alternative hypothesis (Ha) based on the research question or problem.
    2.Choose a significance Level:Select a significane level(α), which represents the maximum probability of rejecting the null hypothesis when it is actually true.
    3.Collect Data: Gather sample data relevant to the research question or problem.
    4.Calculate Test Statistic: Compute a test statistic based on the sample data and the chosen test method. The test statistic quantifies the difference between the sample data and what would be expected under the null hypothesis.
    5.Determine Critical Region: Determine the critical region of the test statistic based on the chosen significance level and the distribution of the test statistic under the null hypothesis.
    6.Make a Decision: Compare the test statistic to the critical region. If the test statistic falls within the critical region, reject the null hypothesis in favor of the alternative hypothesis. If it falls outside the critical region, fail to reject the null hypothesis.
    7.Draw Conclusion: Interpret the results of the hypothesis test in the context of the research question or problem.

Importance of Hypothesis testing:
1.Informed Decision Making: Hypothesis testing provides a systematic framework for making decisions based on data and evidence. It helps researchers and decision-makers draw conclusions about population parameters, which can inform actions and policies.
2.Statistical Inference: Hypothesis testing allows researchers to infer properties of a population based on sample data. By testing hypotheses, researchers can assess the plausibility of various claims and theories.
3.Scientific Validation: In scientific research, hypothesis testing is crucial for validating or refuting hypotheses and theories. It helps ensure that scientific claims are supported by empirical evidence.
4.Quality Control: In fields such as manufacturing and quality control, hypothesis testing is used to assess whether products meet certain standards or specifications. It helps identify issues and make improvements to processes.
5.Risk Management: Hypothesis testing is valuable in assessing risks and uncertainties in various contexts, such as finance, healthcare, and environmental science. It provides a basis for making decisions under uncertainty.
   
# In[10]:


# Qus 4

1.Null Hypothesis (H0): 
    The average weight of male college students is equal to or less than the average weight of female college students.
2.Alternative Hypothesis (Ha): 
    The average weight of male college students is greater than the average weight of female college students.
    
Mathematically , the hypotheses can be expresses as:
    Ho:μ_male<=μ_female
    Ha:μ_male>μ_female
# In[11]:


# Qus 5


# In[15]:


import numpy as np
from scipy.stats import t

def two_sample_t_test(sample1, sample2, alpha=0.05):
    mean1=np.mean(sample1)
    mean2=np.mean(sample2)
    n1=len(sample1)
    n2=len(sample2)
    var1=np.var(sample1,ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    pooled_std=np.sqrt((var1/n1)+(var2/n2))
    
    t_statistic=(mean1-mean2)/pooled_std
    
    df=n1+n2-2
    
    critical_t=t.ppf(1-alpha,df)
    
    p_value=2*(1-t.cdf(abs(t_statistic),df))

    if abs(t_statistic) > critical_t:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")
    
    print("t-statistic:", t_statistic)
    print("Degrees of freedom:", df)
    print("p-value:", p_value)
    
sample1 = np.array([68, 72, 63, 71, 65])  # Sample from population 1
sample2 = np.array([61, 65, 59, 63, 67])  # Sample from population 2
alpha = 0.05  # Significance level

two_sample_t_test(sample1, sample2, alpha)

    


# In[16]:


# Qus 6

Null Hypothesis (H0):
The null hypothesis is a statement of no effect, no difference, or no relationship in the population.
It represents the default assumption or the status quo that researchers seek to test against.
In hypothesis testing, the null hypothesis is typically denoted as H0.
The goal is to either reject or fail to reject the null hypothesis based on the sample data.
Examples:
The mean income of two groups is the same.
There is no difference in test scores between two teaching methods.
A new drug has no effect on blood pressure.

Alternative Hypothesis (Ha):
The alternative hypothesis is a statement that contradicts the null hypothesis.
It represents the researcher's claim or the effect they are interested in investigating.
In hypothesis testing, the alternative hypothesis is denoted as Ha.
The alternative hypothesis provides an alternative explanation to the null hypothesis when there is evidence to reject the null hypothesis.
Examples:
The mean income of two groups is different.
There is a difference in test scores between two teaching methods.
A new drug has an effect on blood pressure.
# In[17]:


# Qus 7

Certainly! Here are the steps involved in hypothesis testing:

1. **Formulate Hypotheses**:
   - State the null hypothesis (H0) and the alternative hypothesis (Ha) based on the research question or problem.
   - H0 typically represents the default assumption or no effect, while Ha represents the researcher's claim or the effect of interest.

2. **Choose a Significance Level**:
   - Select a significance level (α), which represents the maximum probability of rejecting the null hypothesis when it is actually true.
   - Common significance levels include 0.05 (5%) and 0.01 (1%).

3. **Select a Test Statistic**:
   - Choose an appropriate test statistic based on the type of data and the hypotheses being tested.
   - Common test statistics include the t-test, z-test, chi-square test, ANOVA, etc.

4. **Collect Data**:
   - Gather sample data relevant to the research question or problem.
   - Ensure that the data collection process is unbiased and representative of the population.

5. **Calculate the Test Statistic**:
   - Compute the value of the chosen test statistic based on the sample data and the null hypothesis.
   - The test statistic quantifies the difference between the sample data and what would be expected under the null hypothesis.

6. **Determine the Critical Region**:
   - Determine the critical region of the test statistic based on the chosen significance level and the distribution of the test statistic under the null hypothesis.
   - Critical values are obtained from statistical tables or calculated using software.

7. **Make a Decision**:
   - Compare the calculated test statistic to the critical value(s) from the distribution.
   - If the test statistic falls within the critical region, reject the null hypothesis in favor of the alternative hypothesis. If it falls outside the critical region, fail to reject the null hypothesis.

8. **Calculate the P-Value** (optional):
   - Alternatively, calculate the p-value associated with the test statistic.
   - The p-value represents the probability of obtaining a test statistic as extreme as or more extreme than the observed one, assuming the null hypothesis is true.
   - If the p-value is less than the significance level (α), reject the null hypothesis.

9. **Draw Conclusion**:
   - Interpret the results of the hypothesis test in the context of the research question or problem.
   - State whether there is sufficient evidence to reject the null hypothesis and support the alternative hypothesis, or if there is insufficient evidence to do so.

10. **Report Findings**:
    - Clearly communicate the findings of the hypothesis test, including the decision made, the test statistic, the critical value(s) or p-value, and any relevant conclusions or implications.

These steps provide a systematic framework for conducting hypothesis testing and making informed decisions based on sample data and statistical analysis.
# In[18]:


# Qus 8

The p-value, or probability value, is a measure used in hypothesis testing to quantify the strength of evidence against the null hypothesis (H0). It represents the probability of obtaining a test statistic as extreme as or more extreme than the observed one, assuming that the null hypothesis is true.

Calculation of the p-value:
The p-value is calculated based on the observed sample data and the null hypothesis.
It depends on the specific statistical test being used and the chosen test statistic.
For example, in a t-test, the p-value is calculated based on the t-statistic and the degrees of freedom.

Interpretation of the p-value:
A low p-value indicates strong evidence against the null hypothesis.
A high p-value suggests weak evidence against the null hypothesis.
The p-value ranges between 0 and 1. A p-value close to 0 indicates strong evidence against the null hypothesis, while a p-value close to 1 suggests weak evidence against the null hypothesis.

Decision Rule:
In hypothesis testing, the p-value is compared to the chosen significance level (α).
If the p-value is less than or equal to α, the null hypothesis is rejected.
If the p-value is greater than α, the null hypothesis is not rejected.

Significance Level (α):
The significance level represents the threshold for rejecting the null hypothesis.
Commonly used significance levels include 0.05 (5%) and 0.01 (1%).
Researchers choose the significance level based on the desired balance between Type I error (false positive) and Type II error (false negative).

Conclusion:
A small p-value indicates that the observed data is unlikely to have occurred under the null hypothesis, providing evidence in favor of the alternative hypothesis.
A large p-value suggests that the observed data is consistent with the null hypothesis, and there is insufficient evidence to reject it.
The p-value helps researchers make informed decisions about whether to accept or reject the null hypothesis based on the strength of the evidence provided by the sample data.

# In[19]:


# Qus 9


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Generate x values
x = np.linspace(-5, 5, 1000)

# Degrees of freedom
df = 10

# Calculate y values (probability density function)
y = t.pdf(x, df)

# Plot the t-distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='t-distribution with df=10', color='blue')
plt.title("Student's t-Distribution (df=10)")
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


# Qus 10


# In[22]:


import numpy as np
from scipy.stats import ttest_ind

def two_sample_t_test(sample1, sample2, alpha=0.05):
    # Perform two-sample t-test
    t_statistic, p_value = ttest_ind(sample1, sample2)
    
    # Determine if null hypothesis is rejected
    if p_value < alpha:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")
    
    # Print t-statistic and p-value
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

# Generate two random samples of equal size
sample_size = 50
sample1 = np.random.normal(loc=10, scale=2, size=sample_size)
sample2 = np.random.normal(loc=10, scale=2, size=sample_size)

# Null hypothesis: population means are equal
alpha = 0.05  # Significance level

# Perform two-sample t-test
two_sample_t_test(sample1, sample2, alpha)


# In[23]:


# Qus 11

The Student's t-distribution, often referred to simply as the t-distribution, is a probability distribution that arises when estimating the population mean from a sample with a small sample size or when the population standard deviation is unknown. It is similar to the standard normal distribution but has heavier tails, which means it has more probability in the tails and less in the center compared to the normal distribution.
used:
    Sample Size is Small:
        When the sample size is small (typically less than 30) and the population standard deviation is unknown, the t-distribution is used to account for the increased uncertainty in estimating the population mean.

    Population Standard Deviation is Unknown:
        Even for larger sample sizes, if the population standard deviation is unknown and must be estimated from the sample, the t-distribution is used instead of the standard normal distribution.

    Normality Assumption is Violated:
        In some cases, when the population is not normally distributed but the sample size is large enough (usually greater than 30), the t-distribution can still be used due to the central limit theorem.
# In[24]:


# Qus 12

The t-statistic is a measure used in hypothesis testing to assess the significance of the difference between sample means or to test hypotheses about population means when the population standard deviation is unknown and must be estimated from the sample data. It quantifies how many standard errors the sample mean is away from the null hypothesis mean.

formula link:
https://images.search.yahoo.com/images/view;_ylt=AwrjbYmtQCtmCB48Y0uJzbkF;_ylu=c2VjA3NyBHNsawNpbWcEb2lkAzIyZGY2MGNkNTM4NWRkMjBkNGQ5ZjNiNzhkNTA1ZmE3BGdwb3MDNARpdANiaW5n?back=https%3A%2F%2Fimages.search.yahoo.com%2Fsearch%2Fimages%3Fp%3Dt-statistic%2Bformula%2Bt-test%26type%3DE210US885G0%26fr%3Dmcafee%26fr2%3Dpiv-web%26tab%3Dorganic%26ri%3D4&w=1205&h=631&imgurl=microbenotes.com%2Fwp-content%2Fuploads%2F2023%2F08%2FT-Test-Formula.jpeg&rurl=https%3A%2F%2Fsciencesavers.info%2Ft-test-definition-formulation-sorts-functions%2F&size=68.8KB&p=t-statistic+formula+t-test&oid=22df60cd5385dd20d4d9f3b78d505fa7&fr2=piv-web&fr=mcafee&tt=T-test%3A+Definition%2C+Formulation%2C+Sorts%2C+Functions+-+sciencesavers&b=0&ni=21&no=4&ts=&tab=organic&sigr=8DIRXg1HERGd&sigb=YD7y9QMBSzzJ&sigi=aGKpmNTozlHJ&sigt=WdUL5tFb1EN0&.crumb=13Py3iePTPU&fr=mcafee&fr2=piv-web&type=E210US885G0
# In[25]:


# Qus 13


# In[26]:


import numpy as np
from scipy.stats import t

sample_mean=500
sample_std_dev=50
sample_size=50
confidence_level=0.95

degree_of_freedom=sample_size-1
critical_value=t.ppf((1+confidence_level)/2,degree_of_freedom)

margin_of_error=critical_value *(sample_std_dev / np.sqrt(sample_size))

lower_bound=sample_mean-margin_of_error
upper_bound=sample_mean+margin_of_error
print("Lower bound:",lower_bound)
print("Upper bound:",upper_bound)


# In[27]:


# Qus 14


# In[29]:


import numpy as np
from scipy.stats import t

sample_mean=8
sample_std_dev=3
sample_size=100
population_mean_null=10
significance_level=0.05

t_statistic=(sample_mean-population_mean_null)/(sample_std_dev/np.sqrt(sample_size))

degrees_of_freedom=sample_size-1

critical_value=t.ppf(significance_level, degrees_of_freedom)

if t_statistic<critical_value:
    print("Reject the null Hypothesis")
else:
    print("Fail to reject the null hypothesis")
print("t_statistic:",t_statistic)
print("Critical_value:",critical_value)


# In[30]:


# Qus 15


# In[31]:


import numpy as np
from scipy.stats import t

sample_mean=4.8
population_std_dev=0.5
sample_size=25
population_mean_null=5
significance_level=0.01

t_statistic=(sample_mean-population_mean_null)/(population_std_dev/np.sqrt(sample_size))

degrees_of_freedom=sample_size-1

critical_value=t.ppf(significance_level, degrees_of_freedom)

if t_statistic<critical_value:
    print("Reject the null Hypothesis")
else:
    print("Fail to reject the null hypothesis")
print("t_statistic:",t_statistic)
print("Critical_value:",critical_value)


# In[33]:


# Qus 16


# In[34]:


import numpy as np
from scipy.stats import t

n1=30
mean1=80
std_dev1=10

n2=40
mean2=75
std_dev2=8

alpha=.01

pooled_std_dev=np.sqrt(((std_dev1 ** 2)/n1)+((std_dev2 **2)/n2))

t_statistic=(mean1-mean2)/pooled_std_dev

degrees_of_freedom=n1+n2-2

critical_value=t.ppf(1-alpha/2,degrees_of_freedom)

if abs(t_statistic)<critical_value:
    print("Reject the null Hypothesis")
else:
    print("Fail to reject the null hypothesis")
print("t_statistic:",t_statistic)
print("Critical_value:",critical_value)


# In[35]:


# Qus 17


# In[36]:


import numpy as np
from scipy.stats import t

sample_mean=4
sample_std_dev=1.5
sample_size=50
confidence_level=0.99

degree_of_freedom=sample_size-1
critical_value=t.ppf((1+confidence_level)/2,degree_of_freedom)

margin_of_error=critical_value *(sample_std_dev / np.sqrt(sample_size))

lower_bound=sample_mean-margin_of_error
upper_bound=sample_mean+margin_of_error
print("Lower bound:",lower_bound)
print("Upper bound:",upper_bound)


# In[ ]:





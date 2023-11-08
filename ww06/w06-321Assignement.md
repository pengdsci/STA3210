---
title: 'STA321 Week #6 Assignment'
author: "Cheng Peng"
date: "3/3/2021"
output:
  word_document: 
    toc: yes
    toc_depth: 4
    fig_caption: yes
    keep_md: yes
  html_document: 
    toc: no
    toc_depth: 4
    fig_width: 6
    fig_height: 4
    fig_caption: yes
    number_sections: yes
    theme: readable
  pdf_document: 
    toc: no
    toc_depth: 4
    fig_caption: yes
    number_sections: yes
    fig_width: 5
    fig_height: 4
---



<style type="text/css">
h1.title {
  font-size: 20px;
  color: DarkRed;
  text-align: center;
}
h4.author { /* Header 4 - and the author and data headers use this too  */
    font-size: 18px;
  font-family: "Times New Roman", Times, serif;
  color: DarkRed;
  text-align: center;
}
h4.date { /* Header 4 - and the author and data headers use this too  */
  font-size: 18px;
  font-family: "Times New Roman", Times, serif;
  color: DarkBlue;
  text-align: center;
}
h1 { /* Header 3 - and the author and data headers use this too  */
    font-size: 22px;
    font-family: "Times New Roman", Times, serif;
    color: darkred;
    text-align: center;
}
h2 { /* Header 3 - and the author and data headers use this too  */
    font-size: 18px;
    font-family: "Times New Roman", Times, serif;
    color: navy;
    text-align: left;
}

h3 { /* Header 3 - and the author and data headers use this too  */
    font-size: 15px;
    font-family: "Times New Roman", Times, serif;
    color: navy;
    text-align: left;
}

h4 { /* Header 4 - and the author and data headers use this too  */
    font-size: 18px;
    font-family: "Times New Roman", Times, serif;
    color: darkred;
    text-align: left;
}
</style>




This weeks' assignment focuses on the simple linear regression model. Please follow the instructions to pick an appropriate data set for the following three assignments (combining the three individual assignments to complete project #2)

1. Find a data set that contains at least t categorical variables, three numerical variables, and a **binary response variable**. **Please go to D2L Discussion Board to post the information of your data set so your classmates will use your data for the assignment and project #2 (on logistic regression)**.

2. The number of observations is at least 150. This number of observations is slightly large than the one required in the project #1 since we will introduce a machine learning algorithm using the logistic regression model.

3. If you have a data set with no binary response variable but you really want to use it for your assignments, your can dichotomize the response variable in a meaningful way. For example, you have data set with several predictor variable the potentially associated with the response variable GPA. You can dichotomize the continuous GPA in the following

 bin.gpa = 1 if GPA < 2.75
 bin.gpa = 0 if GPA >= 2.75
 
 This dichotomization is meaningful since most graduate schools use 2.75 as the admission cut-off of GPA.
 
4. Choose numerical predictor variable or a **binary** predictor variable to fit a simple logistic regression model as what I did in the case study in my class note as your assignment for this is week.

To be more specific, you need to provide the following key components in your analysis report.

(a) Describe your data set and the variables
(b) Formulate a practically meaningful analytic questions
(c) Perform exploratory data analysis using graphical or numerical approach.
(d) Build a simple logistic regression model
(e) Interpret the regression coefficients from the practical perspective (odds ratio)
(f) study the behavior of the success probability (probability curve and the rate of change in success probability).
 
 
 
 
 
 
 
 

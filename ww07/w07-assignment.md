---
title: 'Week #7 Assignment'
author: 'Due:  Sunday, 11:30 PM, 3/14/21'
date: "3/11/2021"
output:
  word_document: 
    toc: yes
    toc_depth: 4
    fig_caption: yes
    keep_md: yes
  html_document: 
    toc: yes
    toc_depth: 4
    fig_width: 6
    fig_height: 4
    fig_caption: yes
    number_sections: yes
    theme: readable
  pdf_document: 
    toc: yes
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




This week's assignment focuses on the multiple logistic regression modeling using the same data set you used in the previous week. To be more specific, your data set has to meet the following requirements:

* The response variable must be binary. As I mentioned in the previous assignment, you can make a binary reponse variable from a continuous response variable dichotomization.

* At least two continuous predictor variables

* At least two categorical predictor variables 


**Components of the analysis report**

The report should contain the same components as I included in the case study in this week's class note. Please keep in mind that the interpretation of results is VERY important.

  + Description of your data set and variables
  
  + Research questions
  
  + Data management and variable inspection
  
    - variable creation based on existing variables
    
    - variable transformation
    
    - variable discretization
    
    - handling sparse categorical variables
  
  + model building process 
  
    - candidate models
    
    - manual variable selection
    
    - automatic variable selection
    
    - final model identification
    
    - summary the inferential statistics in the final model. 

  + Conclusion and discussion
  
  
**Remarks**: <br>
1. This week's assignment focuses only on the association analysis. <br>
2. Convert the regression coefficients in the final to odds ratio and then provide practical interpretation.<br>
3. The global goodness-of-fit measures (deviance, AIC, etc) in all candidate models should be used only for the purpose of model selection. 




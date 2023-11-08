---
title: 'STA321 Week #5: Report on the Mini-Project on MLR'
author: 'Due: 11:30 PM, Sunday, 02/28/2021'
date: "2/21/2021"
output:
  word_document: 
    toc: yes
    toc_depth: 4
    fig_caption: yes
    keep_md: yes
  html_document: 
    toc_depth: 4
    fig_width: 6
    fig_height: 4
    fig_caption: yes
    theme: readable
  pdf_document: 
    toc_depth: 4
    fig_caption: yes
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






# Introduction

This assignment focuses on bootstrapping the multiple regression model you identified in your last assignment (week #4). You are expected to use both bootstrap sampling methods to find the confidence intervals for regression coefficients.

Please follow what did in the lecture note to do a similar analysis using your data and summarize the bootstrap analysis results. The mini-project report combines the results you obtained last week with this week's bootstrap results. 

The format of the report should have the following components using last and this week's work as building blocks.

* Introduction
  + rationale of the project - research questions
  + description of data - sampling?
  + sufficient information for addressing the research questions
* Data preparation and exploratory analysis
  + create new variables based on existing variables?
  + discretizing continuous variable? combining categories of categorical variables?
* Model building
  + starting with a full model and perform residual analysis
  + identifying violations and finding remedies
  + need model transformations like the Cox-Cox procedure?
  + variable selection? -backward and forward selection, step-wise selection.
  + model selection using goodness-of-fit measures.
  + selection of the final model.
* Bootstrapping the final model
  + bootstrapping records
  + bootstrapping residuals of the final model obtained in the previous model
* Combining the results of regular model and bootstrap results
  + Summarizing the results
  + Correct interpretation of the coefficients of the regression model. 
  + interpretation with caution if your response variable was transformed in a different scale.
* Summary and discussion
  + outline main findings
  + drawbacks and future improvements
  + recommendations on the applications of the model.
  
  










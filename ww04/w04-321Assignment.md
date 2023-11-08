---
title: 'STA321 Week #3 Assignment - MLR'
author: 'Due: 11:30 PM, Sunday, 02/21/2021'
date: "2/16/2021"
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

This assignment focuses the multiple regression model using various techniques you learned from your previous courses. You will use the data set you selected last week. That data set will also be used for next assignment that is based this week's report.

Please study the first three sections of the class note. Your data analysis and write-up should be similar to what did in the case study in section 4 in the note. In fact, Section 4 can be can considered as a standalone statistical report. 

Your analysis and write-up should have all components in Section 4.

* Description of your data set
* Waht is the research questions
* Exploratory analysis on the data set and and prepare the analytic data for the regression
  + create new variable based on existing ones?
  + drop some irrelevant variables based on your judgment?
* initial full model with all relevant variables and conduct residual diagnostic
  + special patterns in residual plots?
  + violation of model assumptions?
* need to transform the response variable with Box-Cox?
* want to transform explanatory variables to improve goodness-of-fit measures? 
  + please feel free to use my code to extract the goodness-of-fit measures. If you forgot the meaning of the goodness-of-fit measures, please check you old textbook or the Ebook that I suggested on the course web page.
  + Several packages have Box-Cox transformation function. The that I used in the case study is from library {MASS}. You can check the help document if you are sure how to use it. 
* build several candidate models and then select the best one as your model.
* summarize the output including residual diagnostic plots
* interpret the regression regression coefficient as I did in the case study.
* conclusions and discussion.


















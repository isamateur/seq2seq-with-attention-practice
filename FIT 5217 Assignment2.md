## FIT 5217 Assignment2

## 1、Introduction

##### Basic introduction 

Given a lemma (the dictionary form of a word) with its part-of-speech, generate a target inflected form.

An example from the English training data:

```
touch   touching   V; V.PTCP;PRS
```

In the training data, all three fields are given. During the test phase, field 2 is omitted. For example:

```
touch  V;V.PTCP;PRS
```

we need to generate target form by MSD(`V;V.PTCP;PRS`)

------

**Task**

Our task is to generate a model through given three levels of training data (low, medium, high), and use this model to predict when only two fields (like all the previous ones) are given target form

##### Data:

The data I mainly selected this time are four languages, namely Turkish, Arabic, basque, northern_sami, they are from the Altaic family-Turkic-Oguz branch, Semito-Hamitic languages, non-Indo-European languages, Ural. Language families, all belong to different language families



## 2、Related Work

## 3、Experiments

### 3.1、baseline methods

Because when I ended this experiment, I still couldn't run the baseline model provided in this job.
So for this basline, I still chose the baseline of the first assignment.

The basic process of the baseline of the first assignment is:

> 1、The first step is to take out the training data set in the order of prototype, feature, and target form.
>
> 2、The second step is to align the prototype of each word with the corresponding targeted form, as shown in the figure below
>
> ``` 
>                 schielen geschielt V.PTCP;PST
>                         Pr St Su
>                         __|schiele|n
>                         ge|schielt|_
> ```
>
> 3、The two words after alignment are decomposed one by one from the front and from the back respectively
>
> ```
> n$ > $												ge$ > $
> en$ > t$											ges$ > s$
> len$ > lt$										gesc$ > sc$
> elen$ > elt$									gesch$ > sch$
> ielen$ > ielt$								geschi$ > schi$
> hielen$ > hielt$							geschie$ > schie$
> chielen$ > chielt$						geschiel$ > schiel$
> schielen$ > schielt$					geschielt$ > schielt$
> ```
>
> 4、Step 4: Store the results according to the rules in the figure above and store them separately according to different features, just like the concept of bag of words (store each word according to feature and store it in a vocabulary one by one, and then go when predicting. Find the most matching data in the vocabulary)
>
> 5、Step 5: Split the words in the text data set one by one, and then split the separated parts, according to the required feature, go in to find the change with the highest probability, and produce the target form

### 3.2、improve



## 4、Evaluation and Error Analysis

After converting all the steps into the form of sequence to sequence, the results obtained have been significantly improved. I show the respective loss and accuracy of the four languages in the figure below.

|   Language    |        Baseline(Accuracy)         |    After Modified (Accuracy)     |
| :-----------: | :-------------------------------: | :------------------------------: |
|    Arabic     |    arabic[task 1/high]: 0.477     |    arabic[task 1/high]: 0.72     |
|    Basque     |     basque[task 1/high]: 0.06     |    basque[task 1/high]: 0.98     |
| Northern-sami | northern-sami[task 1/high]: 0.611 | northern-sami[task 1/high]: 0.78 |
|    Turkish    |    turkish[task 1/high]: 0.729    |    turkish[task 1/high]: 0.76    |

![image-20210604044641899](../../../../../note/Image/image-20210604044641899.png)

![image-20210604044812940](../../../../../note/Image/image-20210604044812940.png)

![image-20210604044922653](../../../../../note/Image/image-20210604044922653.png)

![image-20210604044526517](../../../../../note/Image/image-20210604044526517.png)

According to the accuracy rate obtained, it is obvious that the accuracy rate of some languages has been significantly improved only in the ordinary seq2seq method, such as basque. There is almost no accuracy rate in the basic baseline, but after using seq2seq , The accuracy rate can even reach 98%. It can be seen that this model is very effective in predicting some languages. Except for basque, other languages have improved a lot, except for the limited improvement in Turkish.
(The accuracy of the Turkish language itself is already very high)
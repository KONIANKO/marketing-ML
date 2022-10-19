
for instructions for the setup see below

# NLP Task

## Part 1:

You are contacted by the marketing department. Over the course of the last  year, they received a lot of feedback from their customers about the products the company is selling. The person you are talking to thinks that “AI” can help them to get a better insight into the customers’ opinion, but they do not know what exactly can be done. They would like to know what customers think  about their products, what products they buy, how happy they are with those products and what the issues are. Propose a solution to the marketing department and build a simple prototype to show them in the next meeting what one can do with their data.

**They would like to know:**

1. what customers think about their products, 
2. what products they buy?
3. how happy they are with those products and 
4. what the issues are. 

**Hint 1:** the marketing department would prefer to have a good visualization of 
the insights into the customers’ opinion

**Hint 2:** the marketing department can be impressed with a simple prototype 
that should not necessary take more than 4 hours to build
Data are attached

## Part 2:

Write a short explanation of your solution apart from explaining your choice of 
the technical solution, mention how you would evaluate the performance of 
the system and control its performance over time when it is productive. 

## Part 3:

The marketing department liked your implementation and would like to scale
it for multiple languages i.e., German, Spanish, Indonesian, Hausa, Russian and 
Chinese in which they have a comparable amount of data. What would be your 
answer to the marketing department? How would you implement the 
solution? What are the challenges here?

## Answer Part 2:

Reminder: **They would like to know:**

1. what customers think about their products, 
2. what products they buy?
3. how happy they are with those products and 
4. what the issues are. 

* To figure out what customers think about the company and the products, as well as about thei user experience, I have decided to perform a sentiment analysis on the review texts.
* Since the data is given without sentiment labels, I am not able to train an own supervised learning model or to fine-tune a pre-trained model. It is possible to use the Vader model, which is an unsupervised learning approach. Vader could provide sentiment labels, which could be used for a further training or fine-tuning of a custom model. I have not followed this approach, since I was not able to evaluate the Vader performance in a short time. The training or fine-tuning would make sense, if I would have evidence, that the Vadel labels are satisfiying.
* I decided to use a commonly used pre-trained transformer model, which is trained for english language: 'finiteautomata/bertweet-base-sentiment-analysis'. Like Vader, it is able to perform a sentiment analysis without own sentiment labels. For the prototype, I have decided to trust the model performance. In a further version, the performance of the sentiment classification would have to be examined.
* Then I have filtered the nouns, adjectives and verbs from the review texts and I have created word clouds, showing the most frequently used nouns, verbs and adjectives in positive and negative reviews. I have ignored so called stop words, like articles etc., which have no special meaning. The font size indicates the importance, or frequency of the particular words. I have collected the 20 most important words. For the prototype, I have not created an additional exceptions list of words besides th stopwords for english.
* The idea behind the word clouds on the different parts of speech ist the following:
    * The nouns could indicate, what products the customers buy. If we look into the word clouds for nouns, these products are mostly products related to cars: car, battery, cable, wire, window etc.
    * The adjectives are important to catch the mood of the people. The word 'long' in the adjectives word cloud for negative reviews could indicate, that people are not happy with the delivery time. 'Chinese' could be a hiden indicator, a synonym for a low quality of the products etc.
    * Verbs could give insights, what people expect the company to improve. I have no good real sample here, but if people write things like 'improve', 'check' atc., the company could understand what it could improve in the future. This is especially interesting if considering not only single words, but n-grams, like word pairs or series of words in a more advanced version of the program. Then, it could catch things like 'improve delivery times'.
* Finally, I have made a pie chart with the distribution of positive, neutral and negative reviews to get a picture of how happy or unhappy the customers are in general (at least those, who has wrote a review). With the assumption, that unhappy people who have issues with their products, write reviews more often, than people, who had no problems, we have to be careful with the evaluation of the pie chart. The percentage of a good user experience could be much higher, than displayed in the pie chart. 

#### Mnitoring of the Performance:

* I would implement multiple sentiment classifiers and then I would do a majority vote. An odd number of classifiers would prevent a draw in the vote.
* E.g. if I have 5 classifiers (Vader, finiteautomata/bertweet-base-sentiment-analysis, ...) and 3 of them predict, that the sentiment in a review is positive, I would assign a positive label to that review.

## Answer Part 3:

* It is possible to use a multi-language model. The advantage is, that I could use one model for all reviews.
* There a good multi-language models, but usually one can find better, language-specific models, which are fine-tuned for a special language. Here, the accuracy of the sentiment predictions could increase. But it would be necessary to set up a high number of models, which would need more human and technical ressources.
* There could be a need to have a person in the developers team, who knows these languages to work with them.


# Instructions for the Setup

0. Clone repository
1. go to project folder in a cmd tool
2. virtualenv venv (create virtual environment)
3. .\venv\Scripts\activate (activate virtual environment)
4. pip install -r requirements.txt (install required packages)
5. go to 'src' folder
6. run '''sentiment.py''' with '''python sentiment.py''' cmd command
7. find the plots in the /data/plots directory
# L2 INFO232 Mini-project Solve Xporters traffic volume problem

Université Paris Saclay
Group Segway
Group Member: CHEN Shuangrui, Alcantara Hernandez Ursula, LIU Yongjie, soltani Hicham, XU zhenhai, YACHOUTI	Mouad
Professor: Isabelle Guyon

### Context

    As a young and disruptive entrepreuneur, 
    you have just acquired a small lemonade stand located next to an highway. 
    The previous owner of the stand told you: "I noticed that, 
    on average, 1 car out of 100 stops at my stand".
    Hopefully, during the last years, 
    he also kept track of the hourly number cars that used this highway, 
    and he gladly accepted to give you his precious records.
    Since lemons rot fast and you want to avoid waste, 
    you would like to use this data to train a model that predicts the traffic volume. 
    You also found the hourly meteo records of the last years.
    Your mission, should you decide to accept it, 
    is to predict the number of cars that will pass by 
    at a given date, hour, and additional meteorological informations.
    The dataset contains 59 features and the solution 
    is the number of cars registred in an hour ranging from 0 to 7280.

### Evaluation

    The problem is a regression problem where you have to define a prediction of a highway traffic volume.
    Each sample is about a specific day and hour and is characterized by 58 features 
    (including specification about time and different weather descriptions). 
    The target variable is the number of vehicles travelling on the highway during this hour.

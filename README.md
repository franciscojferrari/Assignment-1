# Assignment 1
This is the code developed and used for the Problem 7 of the 1st assignment for the Machine Learning Advanced course DD2434.

## Installation
Create a python virutal enviroment with either virtualenv or conda and install the requirements.txt

## Usage
```python
from assignment1 import assignment1, plot_scatter

animals_csv = pd.read_csv("zoo.data", header=None, names =  ["animal","hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize","type"])

class_mapping = {
    1:"Mammal",
    2 : "Bird",
    3 : "Reptile",
    4 : "Fish",
    5 : "Amphibian", 
    6 : "Bug", 
    7 : "Invertebrate"
}

animals['type'] = animals['type'].replace(class_mapping)
animals_nolegs = animals.drop(['legs'], axis=1)
#Hot econding of the legs variable
animals_he =  pd.get_dummies(animals, columns=["legs"])

ass1 = assignment1(animals_he, ['type', 'animal'], 7)
```

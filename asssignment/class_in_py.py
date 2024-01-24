class Rectangle:
    class_instanct_one = 5 # this is class attributes, attributed defined outside any method of class
    #__init__ = object initializer, whenever you create an object, the 
    def __init__(self, length=2, breadth = 4):
        # instance attributes are variables tied to particular object, when create object of this class,
        # it's __init_ constructor function is evoked, this instance attribute are variables that are tied to a particular obj of a given class.
        self.length = length
        self.breadth = breadth
    def area(self):
        return self.length  * self.breadth




rect = Rectangle(3,4)
rect2 = Rectangle(4,8)



print('Area of object rect is',rect.area())
print('Area of object rect  2 is',rect2.area())

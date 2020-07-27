# Performance Analysis tools for generic rotors

## Motivations

As an Aerospace Engineer, I've always been fascinated by VTOL aircraft of all kinds, especially small scale drones (since its possible to build those on a budget). However it always irked me to have to either hunt down rotor performance information from the manufacturer or perform a full CFD simulation in Ansys just to get a reasonable ballpark of the performance a certain rotor should have. 

It felt like it should be possible to come up with a tool that, given all the relevant parameters for a  rotor, it could output a reasonable estimate of its performance over a wide range of relevant scenarios. As it happens there are such methods exist! 

One possible solution which I found particularly useful over the years has been utilizing a blade element analysis combined with a simplified-momentum assumption and empirical corrections (If you're interested in the math behind how these work and their relative accuracy/ tradeoffs I reccomend this article: [Comparison between Blade-Element models of propellers
](https://www.researchgate.net/publication/290308462_Comparison_between_Blade-Element_models_of_propellers). A very good implementation walkthrough is given in  [Helicopter Performance, Stability, and Control](https://www.amazon.com/Helicopter-Performance-Stability-Control-Raymond/dp/1575242095) by R. Prouty, at the end of chapter 1. 

The method outlined by Prouty works very well when working with Rotorcraft that are roughly the size of a helicopter and are expected to carry a similar amount of payload, and fly at roughly the same speeds as a Blackhawk for instance. However, once you attempt to use these methods for making predictions for small-scale drones, the results can often be off by a factor of 2 or more!

There are several complex reasons which cause a model that works well for large scale aircraft to break down when working with drones. Dynamic twist has a much greater impact in the performance of drones than in helicopters. Drones typically can also use high camber airfoils and much larger twists angles than helicopters. To name only a few of the differences that are likely to impact the final accuracy.

The objective of this repo was to find a way to translate the methods that work so well for helicopters, into the world of RC-aircraft. It was found that the simplest way of achieving a satisfactory result was by replacing the empirical corrections used in the classical model with learned corrections trained with a Database of Propeller test data and a simple keras Dense neural network.

## Methodology

I coded.

## Results

Hella good.

## Use cases



## Next steps

TODO PROOF READ

In the future it would be very helpful to have rotorcraft performance as a function of Advance Ratio, so that the performance of the drone during forward flight can be gleaned. It should be fairly straight-forward to achieve this, since UIUC also provides test data for different flight conditions and there are established methods for relaxing the assumptions made when determing hover performance with blade element models into a more general.



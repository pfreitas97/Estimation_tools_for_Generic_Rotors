# Performance Analysis tools for generic rotors

## Motivations

As an Aerospace Engineer, I've always been fascinated by VTOL aircraft of all kinds, especially small scale drones (since its possible to build those on a budget). However it always irked me to have to either hunt down rotor performance information from the manufacturer or perform a full CFD simulation in Ansys just to get a reasonable ballpark of the performance a certain rotor should have. 

It felt like it should be possible to come up with a tool that, given all the relevant parameters for a  rotor, it could output a reasonable estimate of its performance over a wide range of relevant scenarios. As it happens there are such methods exist! 

One possible solution which I found particularly useful over the years has been utilizing a blade element analysis combined with a simplified-momentum assumption and empirical corrections (There are many sources to read more about this such as LINKKK). A walkthrough I have found particularly convenient over the years is given in  [Helicopter Performance, Stability, and Control](https://www.amazon.com/Helicopter-Performance-Stability-Control-Raymond/dp/1575242095) by R. Prouty, at the end of chapter 1. 

Indeed this method works fairly well when working with Rotorcraft that are roughly the size of a helicopter and are expected to carry a similar amount of payload, and fly at roughly the same speeds as a Blackhawk for instance. However, once you attempt to use these methods for making predictions for small-scale drones, the results can often be off by a factor of 2 or greater!

There are several complex reasons which cause the model that works so well for large scale aircraft to break down when working with drones. Dynamic twist has a much greater impact in the performance of drones than in helicopters due to the materials used for instance. Drones typically also use high camber airfoils and much larger twists angles than helicopters. To name only a few of the differences.

The objective of this repo was to find a way to translate these methods that work so well for helicopters, into the world of RC-aircraft. It was found that the simplest way of achieving this was replacing the empirical corrections used in the classical model 

## Methodology

I coded.

## Results

Hella good.

## Use cases



## Next steps




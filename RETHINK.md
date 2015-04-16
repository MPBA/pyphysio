Things
======

- What is the sense of computing the feature during the init?
    - The inheritance
    - If we use a method, a classmethod, we have that it is not mandatory to execute the super-class code..
        BUT???
        - Galaxy approach: any problem but in the syntax: Feature(data).value
        - What about using a parameter object containing every parameter? Naah
        - Why not Settings? Because of the eventual parallelism, anything static
        - A dictionary with some defaults would be a good choice, but what if we want to compute the same feature with diverse parameters?
            Instead of using a type class instance we can use a wrapper, a decorator containing the parameters and the type information.
    
- We can use a DataFrame instead of a list of Series, but it is better to have a list because of performance issues
- I remove the HR variants of Mean Median etc as they are redundant and make no sense in a general context

- Think about if the CacheableDataCalc can be the same class as Feature, FFT for example that is a CDC could be a Feature?
    - It can be a signal's value
    - It is not a numeric value
    YES
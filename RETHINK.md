Things
======

- What is the sense of computing the feature during the init?
    - The inheritance
    - If we use a method, a classmethod, we have that it is not mandatory to execute the super-class code..
        BUT???
        - Galaxy approach: any problem but in the syntax: Feature(data).value
        - What about using a parameter object containing every parameter? Ok
        - Why not Settings? Because of the eventual parallelism, anything static
        - A dictionary with some defaults would be a good choice, but what if we want to compute the same feature with diverse parameters?
            Instead of using a type class instance we can use a wrapper, a decorator containing the parameters and the type information.
            WRONG!
                - Abandoned this way, static method raw_compute contains the algorithm, instances of the class are used to store different sets of parameters
    
- General compute function: we can use a DataFrame instead of a list of Series as parameter, but it is better to have a list because of performance issues

- I remove the HR variants of Mean Median etc as they are redundant and make no sense in a general context

- Think about if the CacheableDataCalc can be the same class as Feature, FFT for example that is a CDC could be a Feature?
    - It can be a signal's value
    - It is not a numeric value
    YES so TODoneO: convert+join them
    
- I begin converting the default series from intervals (IBI) to a generic time series
    - Which kind of value??
        - Maybe anyone and then checked inside the tool, inside the feature computation
            - Tool, nice name
    - Always a pandas Series!
    - Excluded:
        *Snippets
        *tests
        *example_data
        *galaxy

- Changed the CDC parameter with a kwargs parameter, this is uncomfortable i fix... semantically changed kwargs to params
    - and added a system that computes an additional hash-key from the used parameters
        - cid is now an instance method as the get, I added params parameter to every method that has to manage a cid

- FOR Biz
    - Why a cache system?
        - The computation of several kind of features, for example some Frequency Domain (FD) Features, needs vary time-taking steps such as Power Spectrum Density estimations or other transformations on the entire data-set. If our pipeline needs to compute more than one feature that requires the same sub-step we have that it will compute the same thing with the same inputs (and so with the same outputs) more than once, wasting cpu time.
        - So the cache controls this computation redundancy and conserves the computed intermediate data to reuse it for the next demands.
        - The cache data is hidden inside the data-set object, allowing a transparent usage with a high degree of freedom using any kind of data structures.
    
- Used __future__.division in the files

- The algorithms work with rr in ms, have to transform to TimeSeries in s
"""Abstract type representing the strategy used to distribute a Healpix map or alm.
If the user wishes to implement it's own, it should be added as an inherited type,
see `RR` as an example.
"""
abstract type Strategy end

"""The `RR` type should be used when creating a "Distributed" type in order to
specify that the data has been distributed according to "Round Robin".
"""
abstract type RR <: Strategy end

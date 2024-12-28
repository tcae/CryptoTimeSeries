using Classify
Classify.verbosity = 3

Classify._test_setpartitions(129601, Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)


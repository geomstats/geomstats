#!/bin/bash

manifold="all"
n_samples=10
operation="all"
while getopts m:n:o: flag
do
    case "${flag}" in
        m) manifold=${OPTARG};;
        n) n_samples=${OPTARG};;
        o) operation=${OPTARG};;
    esac
done
echo "Manifold: $manifold";
echo "N Samples: $n_samples";
echo "Operation: $operation";


python generate_benchmark_params.py
pytest exp/time_exp.py  --benchmark-columns='min, max'  --benchmark-sort='fullname'
pytest log/time_log.py  --benchmark-columns='min, max'  --benchmark-sort='fullname'
pytest dist/time_dist.py --benchmark-columns='min, max'  --benchmark-sort='fullname'
pytest inner_produuct/time_inner_product.py --benchmark-columns='min, max'  --benchmark-sort='fullname'
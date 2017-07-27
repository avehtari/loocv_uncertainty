Experiments for comparing models with LOO

Small sample size corrections

Non-Gaussianity with Dirichlet and Skewned Generalized T

Notes
-----

Compiling models/sgt.stan with rstan Version 2.14.1 failed when using GCCcore/6.2.0
```
Error in compileCode(f, code, language = language, verbose = verbose) : 
  Compilation ERROR, function(s)/method(s) not created! In file included from /home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/rstan/include/rstan/stan_fit.hpp(62),
                 from /home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/rstan/include/rstan/rstaninc.hpp(3),
                 from file415f51d2f167.cpp(576):
/home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/StanHeaders/include/src/stan/optimization/bfgs.hpp(113): warning #858: type qualifier on return type is meaningless
        const size_t iter_num() const { return _itNum; }
        ^

In file included from /home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/rstan/include/rstan/stan_fit.hpp(103),
                 from /home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/rstan/include/rstan/rstaninc.hpp(3),
                 from file415f51d2f167.cpp(576):
/home/tsivula/R/x86_64-pc-linux-gnu-library/3.3/rstan/include/rstan/sum_values.hpp(79): warning #858: type qualifier on return type is meaningless
      const size_t called() 
Calls: capture.output ... stan_model -> cxxfunctionplus -> cxxfunction -> compileCode
In addition: Warning message:
running command '/share/apps/easybuild/software/R/3.3.2-iomkl-triton-2017a/lib64/R/bin/R CMD SHLIB file415f51d2f167.cpp 2> file415f51d2f167.cpp.err.txt' had status 1 
Execution halted
```
Using GCCcore/5.4.0 instead seemed to work

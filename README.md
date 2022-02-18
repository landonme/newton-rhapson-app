# Newton Rhapson App
View deployed application here: https://newton-rhapson.herokuapp.com/

This app uses the Newton Rhapson algorithm built from scratch to find the maximum likelihood estimators of a weibull distribution.

Maximum Liklihood Estimation is a method for estimating parameters of an assumed probability distribution given a set of data. It finds the parameter points that maximize the likelihood of randomly observing the data at hand. To do this, it literally finds the parameters that maximize the likelihood function (often converted to the log-likelihood for simplicity).

In most cases, the first order condition of the likelihood function cannot be solved explicity, and this is where root estimation algorithms like Newton Rhapson come into play. Given a continuous and differentiable function, Newton's method approximates the root by taking a straight line tangent to it, learning from the error, and continuing until convergence.

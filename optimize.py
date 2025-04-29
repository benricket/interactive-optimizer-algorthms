import scipy
import numpy as np
import time

class Optimizer():
    
    def __init__(self):
        self.cost_history = []
        self.x_history = []

    def optimize(self, fun,x0,params):
        """
        Perform optimization on a given cost function from a given guess.
        """
        #print(f"Initial guess: {x0}, value: {fun(x0)}")
        self.cost_history = []
        self.x_history = []
        options = {}
        method = params["method"]
        options["maxiter"] = params.get("max_iters",10000)
        tol = params.get("tol",1e-6)
        if method == "gradient_descent":
            pass
        elif method == "newton":
            x = x0
            val = fun(x)
            stop_attempts = 0
            for i in range(options["maxiter"]):
                # Compute Hessian and gradient

                # This is intentionally slow to demonstrate why approximate
                # algorithms are more useful 
                hess = scipy.linalg.inv(scipy.differentiate.hessian(fun,x).ddf)
                grad = scipy.optimize.approx_fprime(x,fun)

                # Perform Newton's method
                x_new = x - hess @ grad
                val_new = fun(x_new)
                self.callback_newton(x_new,val_new)
                
                if np.abs(val_new - val) < tol:
                    stop_attempts += 1
                    if stop_attempts > 2:
                        break
                else:
                    stop_attempts = 0

                # Update guess
                x = x_new
                val = val_new

        elif method == "newton-cg":
            pass
        elif method == "BFGS":
            ans = scipy.optimize.minimize(fun,x0,options=options,method="BFGS",tol=tol,callback=self.callback)
            print(f"iters: {ans.nit}")
            #print(f"ans: {ans.x}")
        else:
            #print("method not found")
            pass

    def callback(self, x, *args):
        """
        Callback function to add guess to history
        """
        #self.cost_history.append(args.fun)
        self.x_history.append(x)
        #print(f"x: {x}, cost: {cost(x)}")

    def callback_newton(self,x,val):
        """
        Callback function to add guess to history
        """
        self.cost_history.append(val)
        self.x_history.append(x)
        #print(f"x: {x}, cost: {cost(x)}")

def rosenbrock(x):
    return scipy.optimize.rosen(x)

if __name__ == "__main__":
    opt = Optimizer()
    guess = (10*np.random.rand(1,20)[0]).tolist()
    #print('\n\n\n\n\n')
    t0 = time.time()
    opt.optimize(rosenbrock,guess,params={"method":"BFGS"})
    print(f"Time: {time.time() - t0}")
    


